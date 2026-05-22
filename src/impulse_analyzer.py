import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
#import warnings

# --- Metadata del software (IEC 61083-2 Sec. 7) ------------------------------------------------------------------------
__app_name__ = "Analizador de Impulsos atmosféricos tipo rayo (1.2/50 us)."
__version__ = "1.0.0"
__release_date__ = "2026-03-24"
__algorithms_supported__ = ["Full Lightning Impulse (LI)", "Chopped Lightning Impulse (LIC)"]
__parameters_validated__ = ["Valor Pico (Ut)", "Tiempo de Frente (T1)", "Tiempo de Cola (T2)", "Sobrepasamiento (OS)"]
# -----------------------------------------------------------------------------------------------------------------------

class LightningImpulseAnalyzer:
    def __init__(self, voltage_data, sampling_period, sigma_fit):
        # Validaciones de seguridad:
        if sampling_period <= 0:
            raise ValueError("Error: El periodo de muestreo debe ser mayor a 0.")
        self.sampling_period = sampling_period

        if sigma_fit <= 0:
            #warnings.warn("sigma_fit debe ser mayor a 0. Se adoptará el valor por defecto = 0.1.")
            self.sigma_fit = 0.1
        else:
            self.sigma_fit = sigma_fit

        # Datos de entrada:
        self.raw_voltage = np.array(voltage_data)

        # Generar array de tiempo: t = index * intervalo.
        self.time_axis = np.arange(len(self.raw_voltage)) * self.sampling_period

        # Tipo de onda:
        self.impulse_type = None

        # Parámetros calculados:
        self.idx_peak = None
        self.time_star_impulse = None
        self.offset_value = None
        self.zeroed_curve = None
        self.zeroed_curve_abs = None
        self.peak_value = None
        self.Ue = None
        self.polarity = None
        self.factor = None
        self.norm_voltage = None
        self.start_slice = None
        self.end_slice = None
        self.fit_voltage = None
        self.fit_time = None
        self.fitted_params = None
        self.fitted_curve = None
        self.base_curve = None
        self.Ub = None
        self.residual_curve = None
        self.filter_coeffs = None
        self.filtered_residual = None
        self.test_voltage_curve_abs = None
        self.test_voltage_curve = None
        self.test_voltage_curve_norm = None
        self.Ut = None
        self.idx_peak_Ut = None
        self.Tcutting_moment = None
        self.idx_deviation = None
        self.t_L = None
        self.aligned_time_axis = None
        self.E = None
        self.chopped_front_voltage = None
        self.chopped_front_time = None
        self.U_collapse = None
        self.results = {
            "Ut": None,
            "T1": None,
            "T2": None,
            "OS": None
        }

    def _remove_offset(self):
        # a) Encontrar el nivel base de la curva registrada.
        self.idx_peak = np.argmax(np.abs(self.raw_voltage))
        front = self.raw_voltage[:self.idx_peak]
        n = int(self.idx_peak * 0.3)

        if n < 1:
            raise ValueError("Error: No hay suficientes muestras de pre-trigger para calcular el offset. Ajuste el delay o el nivel de trigger, y vuelva a intentarlo.")

        pre_trigger = self.raw_voltage[:n]
        mean = np.mean(pre_trigger)
        std = np.std(pre_trigger)

        diff = np.abs(front - mean)
        max_index = np.flatnonzero(diff >= 5 * std)[0]

        diff_2 = np.abs(front[:max_index] - mean)
        valid_index = np.flatnonzero(diff_2 <= std)

        if valid_index.size > 0:
            idx = valid_index[-1]
            background_noise = self.raw_voltage[:idx]
        else:
            idx = 0
            background_noise = self.raw_voltage[0:1]

        self.time_star_impulse = self.time_axis[idx]
        self.offset_value = np.mean(background_noise)

        # b) Eliminar el offset de la curva registrada.
        self.zeroed_curve = self.raw_voltage - self.offset_value

    def _polarity_normalization(self):
        if self.zeroed_curve is None:
            raise ValueError("Error: Falta quitar el offset de la onda.")

        # c) Encontrar el valor extremo, Ue, de la curva registrada compensada en offset, U0(t).
        # Encontrar el índice del máximo valor absoluto.
        self.Ue = self.zeroed_curve[self.idx_peak]

        # Determinar la polaridad de la señal.
        if self.Ue >= 0:
            self.polarity = "Positiva"
            self.factor = 1.0
        else:
            self.polarity = "Negativa"
            self.factor = -1.0

        self.zeroed_curve_abs = self.zeroed_curve * self.factor
        self.peak_value = self.zeroed_curve_abs[self.idx_peak]

    def _normalize_waveform(self):
        if self.zeroed_curve_abs is None:
            raise ValueError("Error: Falta normalizar la polaridad de la onda.")

        self.norm_voltage = self.zeroed_curve_abs / self.peak_value

    @staticmethod
    def _find_limit_index(v_array, threshold, mode):
        if mode == "front":
            # Invierte el array para que vaya en sentido decreciente.
            front_reversed = v_array[::-1]

            # Buscar el primer valor que sea menor que el umbral.
            idx_reversed = np.argmax(front_reversed < threshold)

            if idx_reversed == 0 and front_reversed[0] >= threshold:
                raise ValueError("Error: No se encontraron datos bajo el umbral en el frente.")

            # Convertir el índice invertido al índice original del segmento.
            idx = (len(v_array) - 1) - idx_reversed
        elif mode == "tail":
            idx = np.argmax(v_array < threshold)

            # Validación:
            if not np.any(v_array < threshold):
                raise ValueError("Error: La señal no cae por debajo del umbral en la cola.")
        else:
            raise ValueError("Modo desconocido. Use 'front' o 'tail'.")
        return idx

    def _cutting_signal(self):
        if self.impulse_type == "chopped":
            raise RuntimeError("Este método no aplica a impulsos cortados.")

        if self.peak_value is None:
            raise ValueError("Error: Falta normalizar la onda.")

        front_data = self.zeroed_curve_abs[:self.idx_peak]
        tail_data = self.zeroed_curve_abs[self.idx_peak:]

        threshold_20 = 0.2 * self.peak_value
        threshold_40 = 0.4 * self.peak_value

        # d) Encontrar la última muestra en el frente, inferior a 0,2 * Ue.
        try:
            idx_20 = self._find_limit_index(front_data, threshold_20, mode="front")
        except ValueError:
            raise ValueError("No se encontraron suficientes datos en el frente de la onda (umbral 20%). Verifique la escala de tiempo o si hay ruido excesivo.")

        # e) Encontrar la última muestra en la cola, superior a 0,4 * Ue.
        try:
            idx_40_local = self._find_limit_index(tail_data, threshold_40, mode="tail")
        except ValueError:
            raise ValueError("La onda no decae al 40% del pico en la cola para realizar el ajuste de curva. Aumente la escala de tiempo (Time/DIV).")
        idx_40 = self.idx_peak + idx_40_local

        # Limites inferior y superior:
        self.start_slice = idx_20 + 1
        self.end_slice = idx_40 + 1

        self.fit_voltage = self.zeroed_curve_abs[self.start_slice:self.end_slice]
        self.fit_time = self.time_axis[self.start_slice:self.end_slice]

    @staticmethod
    def _double_exponential_func(t, U, tau1, tau2, td):
        dt = t - td
        # Máscara de seguridad. Para evitar que diverja el ajuste de curva.
        mask = dt >= 0
        # Crear el contenedor de resultados.
        result = np.zeros_like(dt, dtype=np.float64)
        # Calcular solo sobre los tiempos válidos (positivos).
        valid_dt = dt[mask]
        val = U * (np.exp(-valid_dt / tau1) - np.exp(-valid_dt / tau2))
        # Insertar los valores calculados.
        result[mask] = val
        return result

    def _fit_base_curve(self):
        # g) Ajustar la función de doble exponencial a los datos recortados.
        if self.impulse_type == "chopped":
            raise RuntimeError("Este método no aplica a impulsos cortados.")

        if self.fit_voltage is None or self.fit_time is None:
            raise ValueError("Error: Falta segmentar los datos de la onda para el ajuste.")

        # Estimación de parámetros iniciales.
        p0_U = self.peak_value
        p0_tau1 = 70e-6
        p0_tau2 = 0.4e-6
        p0_td = self.fit_time[0]
        initial_guess = [p0_U, p0_tau1, p0_tau2, p0_td]

        idx_peak_in_slice = np.argmax(np.abs(self.fit_voltage))
        sigma = np.ones_like(self.fit_voltage)
        # sigma varía entre 0 y 1. Mientras menor sea sigma, el ajuste en el frente es más preciso.
        sigma[:idx_peak_in_slice + 5] = self.sigma_fit

        # Ejecutar el ajuste de curva (Levenberg-Marquardt).
        try:
            popt, pcov = curve_fit(
                self._double_exponential_func,
                self.fit_time,
                self.fit_voltage,
                p0 = initial_guess,
                sigma = sigma,
                absolute_sigma = False,
                maxfev = 100000
            )
            self.fitted_params = {
                'U': popt[0],
                'tau1': popt[1],
                'tau2': popt[2],
                'td': popt[3]
            }

        except RuntimeError as e:
            raise ValueError("Falló el ajuste matemático de la curva base. La forma de onda puede estar muy distorsionada o cortada prematuramente.")

        # Generar la función ajustada con los parámetros encontrados.
        self.fitted_curve = self._double_exponential_func(self.fit_time,
                                                          self.fitted_params['U'],
                                                          self.fitted_params['tau1'],
                                                          self.fitted_params['tau2'],
                                                          self.fitted_params['td'])

    def _construct_base_curve(self):
        # h) Construir la curva base Um(t).
        if self.impulse_type == "chopped":
            raise RuntimeError("Este método no aplica a impulsos cortados.")

        if self.fitted_params is None:
            raise ValueError("Error: Falta ejecutar fit_base_curve.")

        self.base_curve = self._double_exponential_func(self.time_axis,
                                                        self.fitted_params['U'],
                                                        self.fitted_params['tau1'],
                                                        self.fitted_params['tau2'],
                                                        self.fitted_params['td'])

        # n) Determinar el máximo de la curva base (Ub).
        self.Ub = np.max(self.base_curve)

    def _calculate_residual_curve(self):
        # i) Obtener la curva residual: R(t) = U0(t) - Um(t).
        if self.zeroed_curve_abs is None:
            raise ValueError("Error: Falta quitar el offset de la onda.")
        if self.base_curve is None:
            raise ValueError("Error: Falta encontrar la curva base.")

        self.residual_curve = self.zeroed_curve_abs - self.base_curve

    def _create_digital_filter(self):
        # j) Crear el filtro digital.
        # Constante dada por la norma IEC 60060-1 para el diseño del filtro.
        d = 2.2e-12

        # Cálculo de la constante intermedia c.
        c = np.tan((np.pi * self.sampling_period) / np.sqrt(d))

        # Cálculo de coeficientes del filtro.
        b0 = c / (1 + c)
        b1 = b0
        a1 = (1 - c) / (1 + c)

        # La norma plantea la ecuación recursiva:
        # y(i) = b0*x(i) + b1*x(i-1) + a1*y(i-1)
        # Scipy 'filtfilt' usa la forma:
        # a0*y[n] + a1*y[n-1] = b0*x[n] + b1*x[n-1]

        b = np.array([b0, b1])
        a = np.array([1.0, -a1])

        return b, a

    def _filter_to_residual(self):
        # k) Aplicar el filtro digital a la curva residual R(t).
        if self.residual_curve is None:
            raise ValueError("Error: Falta calcular la curva residual.")

        # Calcular los coeficientes del filtro.
        b, a = self._create_digital_filter()

        # Aplicar el filtro de fase cero, para obtener la curva residual filtrada Rf(t).
        self.filtered_residual = signal.filtfilt(b, a, self.residual_curve)

    def _construct_test_voltage_curve(self):
        # l) Obtener la curva de tensión de ensayo Ut(t) = Um(t) + Rf(t).
        # Validaciones de estado.
        if self.base_curve is None:
            raise ValueError("Error: Falta calcular la curva base Um(t).")
        if self.filtered_residual is None:
            raise ValueError("Error: Falta calcular la curva residual filtrada Rf(t).")

        self.test_voltage_curve_abs = self.base_curve + self.filtered_residual

        # Devolver signo a la curva:
        self.Ut = np.max(self.test_voltage_curve_abs) * self.factor
        self.test_voltage_curve = self.test_voltage_curve_abs * self.factor
        self.results["Ut"] = self.Ut

        # Curva de tensión de ensayo normalizada a 1.
        self.test_voltage_curve_norm = self.test_voltage_curve / np.abs(self.Ut)

    @staticmethod
    def _linear_interpolation(t_array, v_array, idx_low, target_voltage):
        v1 = v_array[idx_low]
        t1 = t_array[idx_low]

        if idx_low + 1 < len(v_array) and v_array[idx_low + 1] > v1:
            # Indice siguiente.
            idx_high = idx_low + 1
        elif idx_low - 1 >= 0 and v_array[idx_low - 1] > v1:
            # Indice anterior.
            idx_high = idx_low - 1
        else:
            # Caso borde o valor exacto.
            return t1

        v2 = v_array[idx_high]
        t2 = t_array[idx_high]

        # Fórmula: t = t1 + (V_target - V1) * (dt / dV)
        return t1 + (target_voltage - v1) * ((t2 - t1) / (v2 - v1))

    def _calc_front_parameters(self, Ut):
        # Calcular Tiempo de Frente (T1).
        front_v = self.test_voltage_curve_abs[:self.idx_peak_Ut]
        front_t = self.time_axis[:self.idx_peak_Ut]
        v30 = 0.3 * Ut
        v90 = 0.9 * Ut
        idx_30 = self._find_limit_index(front_v, v30, mode="front")
        idx_90 = self._find_limit_index(front_v, v90, mode="front")
        t30 = self._linear_interpolation(front_t, front_v, idx_30, v30)
        t90 = self._linear_interpolation(front_t, front_v, idx_90, v90)

        Tab = t90 - t30
        T1 = Tab / 0.6

        # Calcular Origen Virtual (O1).
        O1 = t30 - 0.5 * Tab
        return O1, T1

    def _calc_tail_parameter(self, Ut, O1):
        # Calcular Tiempo cola (T2).
        tail_v = self.test_voltage_curve_abs[self.idx_peak_Ut:]
        tail_t = self.time_axis[self.idx_peak_Ut:]
        v50 = 0.5 * Ut
        idx_50 = self._find_limit_index(tail_v, v50, mode="tail")
        t50 = self._linear_interpolation(tail_t, tail_v, idx_50, v50)
        T2 = t50 - O1
        return T2

    def _calculate_parameters(self):
        if self.test_voltage_curve_abs is None:
            raise ValueError("Error: Falta calcular la curva de tensión de prueba.")

        Ut = np.abs(self.Ut)
        self.idx_peak_Ut = np.argmax(self.test_voltage_curve_abs)
        self.results["OS"] = 100 * (self.peak_value - self.Ub) / self.peak_value

        # Calcular O1 y T1.
        try:
            O1, T1 = self._calc_front_parameters(Ut)
            self.results["T1"] = T1
        except ValueError:
            raise ValueError("No se pudo calcular el Tiempo de Frente (T1). El frente de onda puede tener demasiado ruido o no alcanza los niveles del 30% y 90%.")

        # Calcular T2.
        try:
            if self.impulse_type == "full":
                T2 = self._calc_tail_parameter(Ut, O1)
            elif self.impulse_type == "chopped":
                T2 = self.Tcutting_moment - O1
            else:
                raise ValueError("Tipo de impulso desconocido.")
            self.results["T2"] = T2
        except ValueError:
            raise ValueError("La onda no decae al 50% dentro de la ventana de captura. Verifique la escala de tiempo (Time/DIV) o el circuito de descarga.")

    def _find_time_lag(self, ref_analyzer):
        levels = [0.3, 0.5, 0.8]
        t_diffs = []

        for level in levels:
            # Tiempos en la señal cortada.
            front_self = self.norm_voltage[:self.idx_peak]
            idx_self = self._find_limit_index(front_self, level, mode="front")
            t_self = self._linear_interpolation(self.time_axis, self.norm_voltage, idx_self, level)

            # Tiempos en la señal de referencia.
            front_ref = ref_analyzer.norm_voltage[:ref_analyzer.idx_peak]
            idx_ref = self._find_limit_index(front_ref, level, mode="front")
            t_ref = self._linear_interpolation(ref_analyzer.time_axis, ref_analyzer.norm_voltage, idx_ref, level)

            # Diferencia para este nivel.
            t_diffs.append(t_ref - t_self)

        # El retraso t_L es el promedio de los desplazamientos.
        self.t_L = np.mean(t_diffs)

    def _adjust_time_lag(self):
        if self.t_L is None:
            raise ValueError("Falta calcular el desfase de tiempo t_L.")

        self.aligned_time_axis = self.time_axis + self.t_L

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Métodos específicos para el análisis de impulsos cortados en la cola,
# que requieren comparación con un impulso completo de referencia.

    def _find_deviation_point(self, ref_analyzer, threshold=0.02):
        if self.norm_voltage is None or ref_analyzer.norm_voltage is None:
            raise ValueError("Las señales deben estar normalizadas.")
        if self.aligned_time_axis is None:
            raise ValueError("Falta alinear el eje de tiempo con t_L primero.")

        # Interpolar la señal cortada al eje de la referencia (tiempo ya alineado).
        chopped_norm_interp = np.interp(
            ref_analyzer.time_axis,
            self.aligned_time_axis,
            self.norm_voltage,
            left=0.0, right=0.0
        )

        # Calcular diferencia absoluta post-pico.
        tail_self = chopped_norm_interp[ref_analyzer.idx_peak:]
        tail_ref = ref_analyzer.norm_voltage[ref_analyzer.idx_peak:]
        diff = np.abs(tail_self - tail_ref)

        # Encontrar el umbral.
        mask_dev = diff > threshold
        if mask_dev.any():
            self.impulse_type = "chopped"
        else:
            self.impulse_type = "full"
            return

        idx_dev_local = np.argmax(mask_dev)
        self.idx_deviation = ref_analyzer.idx_peak + idx_dev_local

    def _select_data_up_to_deviation(self):
        if self.idx_deviation is None:
            raise ValueError("Falta calcular el punto de desviación.")

        # El +1 asegura que el punto de desviación esté incluido en el análisis.
        idx_end = self.idx_deviation + 1

        # .copy() asegura que los datos sean independientes del array original.
        self.chopped_front_voltage = self.norm_voltage[:idx_end].copy()
        self.chopped_front_time = self.time_axis[:idx_end].copy()

    def _find_amplitude_ratio(self, ref_analyzer):
        if self.aligned_time_axis is None:
            raise ValueError("Falta alinear el eje de tiempo.")

        # Encontrar los índices del 30% y 80% de la onda normalizada.
        front_self_norm = self.norm_voltage[:self.idx_peak]
        idx_30 = self._find_limit_index(front_self_norm, 0.3, mode="front")
        idx_80 = self._find_limit_index(front_self_norm, 0.8, mode="front")

        # Extraer los valores de tensión reales (absolutos) en esa zona.
        vals_self = self.zeroed_curve_abs[idx_30:idx_80]
        t_interval = self.aligned_time_axis[idx_30:idx_80]

        # Interpolar la referencia usando sus tensiones reales (absolutas).
        vals_ref = np.interp(
            t_interval,
            ref_analyzer.time_axis,
            ref_analyzer.zeroed_curve_abs
        )

        # Calcular la relación de las amplitudes.
        self.E = np.mean(vals_self) / np.mean(vals_ref)

    def _scale_base_curve(self, ref_analyzer):
        # Extraer los parámetros de la curva de referencia.
        U_ref = ref_analyzer.fitted_params['U']
        tau1 = ref_analyzer.fitted_params['tau1']
        tau2 = ref_analyzer.fitted_params['tau2']
        td = ref_analyzer.fitted_params['td']

        # Escalar la onda de referencia con el factor E.
        U_scaled = U_ref * self.E

        # Construir la nueva curva base evaluando en el tiempo alineado.
        self.base_curve = self._double_exponential_func(
            self.aligned_time_axis,
            U_scaled,
            tau1,
            tau2,
            td
        )
        self.Ub = np.max(self.base_curve)

    def _find_chopping_instant(self):
        if self.norm_voltage is None:
            raise ValueError("Error: La señal no ha sido normalizada.")

        # Buscar el punto de caída más abrupto en la cola usando el Gradiente (dV/dt).
        tail_vals = self.norm_voltage[self.idx_peak:]
        gradient = np.gradient(tail_vals)
        idx_steepest_local = np.argmin(gradient)

        # Encontrar el voltaje inmediatamente antes del colapso (U_collapse).
        grad_2 = np.gradient(gradient[:idx_steepest_local])
        idx_knee_local = np.argmin(grad_2)
        idx_collapse = self.idx_peak + idx_knee_local

        U_collapse = self.zeroed_curve_abs[idx_collapse]

        # Definir puntos C (70%) y D (10%) referidos a U_collapse.
        v70 = 0.7 * U_collapse
        v10 = 0.1 * U_collapse
        fall_segment = self.zeroed_curve_abs[idx_collapse:]
        idx_70_local = np.argmax(fall_segment < v70)
        idx_10_local = np.argmax(fall_segment < v10)

        if idx_70_local == 0 or idx_10_local == 0:
            raise ValueError("Error: No se pudo determinar el 70% o 10% del corte.")

        idx_C = idx_collapse + idx_70_local
        idx_D = idx_collapse + idx_10_local

        # Regresión lineal entre C y D para hallar el instante de corte.
        t_C, v_C = self.time_axis[idx_C], self.zeroed_curve_abs[idx_C]
        t_D, v_D = self.time_axis[idx_D], self.zeroed_curve_abs[idx_D]

        if t_D == t_C:
            raise ValueError("Error: t_C = t_D; no se puede calcular la pendiente de corte.")

        slope = (v_D - v_C) / (t_D - t_C)
        self.Tcutting_moment = t_C + (U_collapse - v_C) / slope
        self.U_collapse = U_collapse

# --------------------------------------------------------------------------------------------------------------------------------------------------------
# Métodos de análisis completo para cada tipo de onda.
    def ref_lightning_impulse(self):
        self.impulse_type = "full"
        # Ejecutar pipeline completo.
        self._remove_offset()
        self._polarity_normalization()
        self._normalize_waveform()
        #-----------------------------------------------------
        self._cutting_signal()
        self._fit_base_curve()
        self._construct_base_curve()
        #-----------------------------------------------------
        self._calculate_residual_curve()
        self._filter_to_residual()
        self._construct_test_voltage_curve()
        self._calculate_parameters()

    def lightning_impulse(self, ref_analyzer):
        if ref_analyzer is None:
            raise ValueError("Error: Falta guardar el impulso de referencia a tensión reducida.")

        self._remove_offset()
        self._polarity_normalization()
        self._normalize_waveform()
        #-----------------------------------------------------
        self._find_time_lag(ref_analyzer)
        self._adjust_time_lag()
        self._find_deviation_point(ref_analyzer)
        if self.impulse_type == "full":
            self._cutting_signal()
            self._fit_base_curve()
            self._construct_base_curve()
        elif self.impulse_type == "chopped":
            self._select_data_up_to_deviation()
            self._find_amplitude_ratio(ref_analyzer)
            self._scale_base_curve(ref_analyzer)
            self._find_chopping_instant()
        #-----------------------------------------------------
        self._calculate_residual_curve()
        self._filter_to_residual()
        self._construct_test_voltage_curve()
        self._calculate_parameters()