import time
from struct import unpack
import sys
import pyvisa
import numpy as np

class GWInstekGDS1000AU:
    # Constante de cuantización vertical del ADC, específica del modelo.
    ADC_STEPS_PER_DIV = 25.0

    def __init__(self):
        self.dso = None
        self.rm = pyvisa.ResourceManager('@py')
        instrument_list = self.rm.list_resources()
        print("Instrumentos encontrados:", instrument_list)
        if instrument_list:
            self.connect(instrument_list[0])

    def connect(self, resource_name):
        try:
            self.dso = self.rm.open_resource(resource_name)
            self.dso.read_termination = '\n'
            self.dso.write_termination = '\n'
            idn = self.dso.query('*IDN?')
            print("Instrumento conectado satisfactoriamente:", idn)
        except Exception as e:
            print("Error al iniciar con el instrumento:", e)
            self.close()

    def get_block_data(self, channel):
        try:
            v_div = self.get_channel_scale(channel)
            self.dso.write(f':acquire{channel}:state?')
            state = self.dso.read()
            if state[0] == '1':
                time.sleep(0.1)
                self.dso.write(f':acquire{channel}:memory?')
                inBuffer = self.dso.read_bytes(10)
                length = len(inBuffer)
                headerlen = 2 + int(chr(inBuffer[1]))
                pkg_length = int(inBuffer[2:headerlen]) + headerlen
                pkg_length = pkg_length - length
                while True:
                    if pkg_length == 0:
                        break
                    else:
                        if pkg_length > 100000:
                            length = 100000
                        else:
                            length = pkg_length
                        try:
                            buf = self.dso.read_bytes(length)
                        except:
                            print('Error al recibir datos del instrumento!')
                            self.close()
                            sys.exit(0)
                        num = len(buf)
                        inBuffer += buf
                        pkg_length = pkg_length - num
                waveform, dt = self.unpack_waveform(inBuffer, headerlen, v_div)
                return inBuffer, waveform, dt
            else:
                print('Error: Forma de onda aún no está lista.')
                return None, None, None
        except Exception as e:
            print("Error al obtener datos:", e)
            self.close()
            return None, None, None

    def unpack_waveform(self, inBuffer, headerlen, vdiv):
        print(inBuffer[:headerlen])
        dt = unpack('>f', inBuffer[headerlen : headerlen + 4])[0]
        print(f'Periodo de muestreo = {dt*1e9:.0f} [ns]')
        waveform_raw = unpack('>%sh' % (int(len(inBuffer[headerlen + 8:]) / 2)), inBuffer[headerlen + 8:])
        waveform_raw = np.array(waveform_raw)
        num = len(waveform_raw)
        print(f'Cantidad de muestras = {num}')
        waveform = waveform_raw * vdiv / self.ADC_STEPS_PER_DIV
        return waveform, dt

    def default_settings(self):
        try:
            self.dso.write('*RST')
            print("Se restableció el instrumento a la configuración de fábrica exitosamente.")
        except Exception as e:
            print("Error al restablecer el instrumento:", e)
            self.close()

    def get_setting(self):
        try:
            current_setting = self.dso.query('*LRN?')
            print(f"Configuracion actual: {current_setting}")
        except Exception as e:
            print("Error al consultar configuración:", e)
            self.close()

    def get_channel_scale(self, channel):
        try:
            scale = self.dso.query(f':channel{channel}:scale?')
            v_scale = float(scale)
            print(f'Escala vertical canal {channel}: {v_scale:.2f} [V/div]')
            return v_scale
        except Exception as e:
            print("Error al obtener la escala vertical:", e)
            self.close()

    def set_channel_scale(self, channel, value):
        try:
            self.dso.write(f':channel{channel}:scale {value}')
            v_scale = self.get_channel_scale(channel)
            if v_scale == value:
                print(f'Escala vertical canal {channel} configurada a: {v_scale:.2f} [V/div]')
            else:
                print('No se pudo configurar la escala vertical.')
        except Exception as e:
            print("Error al configurar la escala vertical:", e)
            self.close()

    def get_timebase_scale(self):
        try:
            scale = self.dso.query(':timebase:scale?')
            h_scale = float(scale)
            print(f'Escala horizontal: {h_scale} [s/div]')
            return h_scale
        except Exception as e:
            print("Error al obtener la escala horizontal:", e)
            self.close()

    def set_timebase_scale(self, value):
        try:
            self.dso.write(f':timebase:scale {value}')
            h_scale = self.get_timebase_scale()
            if h_scale == value:
                print(f'Escala Horizontal configurada a: {h_scale} [s/div]')
            else:
                print('No se pudo configurar la escala horizontal.')
        except Exception as e:
            print("Error al configurar la escala horizontal:", e)
            self.close()

    def get_timebase_position(self):
        try:
            position = float(self.dso.query(':timebase:delay?'))
            print(f'Posición horizontal: {position} [s]')
            return position
        except Exception as e:
            print("Error al obtener la posición horizontal:", e)
            self.close()

    def set_timebase_position(self, value):
        try:
            self.dso.write(f':timebase:delay {value}')
            position = self.get_timebase_position()
            if position == value:
                print(f'Posición horizontal configurada a: {position} [s]')
            else:
                print('No se pudo configurar la posición horizontal.')
        except Exception as e:
            print("Error al configurar la posición horizontal:", e)
            self.close()

    def set_trigger(self, trigger_mode):
        try:
            self.dso.write(trigger_mode)
            print("Modo de disparo configurado exitosamente.")
        except Exception as e:
            print("Error al configurar el modo de disparo:", e)
            self.close()

    def get_trigger_level(self):
        try:
            level = float(self.dso.query(':trigger:level?'))
            return level
        except Exception as e:
            print("Error al obtener el nivel de disparo:", e)
            self.close()

    def set_trigger_level(self, trigger_level):
        try:
            self.dso.write(f':trigger:level {trigger_level}')
            if self.get_trigger_level() == trigger_level:
                print(f"Nivel de disparo configurado: {trigger_level}")
            else:
                print("No se pudo configurar el nivel de disparo.")
        except Exception as e:
            print("Error al configurar el nivel de disparo:", e)
            self.close()

    def get_trigger_coupling(self):
        couplings = ('AC', 'DC')
        try:
            coupling = self.dso.query(':trigger:couple?')
            print(f'Acoplamiento de trigger: {couplings[int(coupling)]}')
            return couplings[int(coupling)]
        except Exception as e:
            print("Error al obtener el acoplamiento de trigger:", e)
            self.close()

    def set_trigger_coupling(self, coupling):
        couplings = ('AC', 'DC')
        try:
            self.dso.write(f':trigger:couple {coupling}')
            current_coupling = self.get_trigger_coupling()
            if current_coupling == couplings[coupling]:
                print(f'Acoplamiento de trigger configurado a: {current_coupling}')
            else:
                print('No se pudo configurar el acoplamiento de trigger.')
        except Exception as e:
            print("Error al configurar el acoplamiento de trigger:", e)
            self.close()

    def get_trigger_mode(self):
        modes = ('Auto', 'Normal')
        try:
            mode = self.dso.query(':trigger:mode?')
            print(f'Modo de trigger: {modes[int(mode)-1]}')
            return modes[int(mode)-1]
        except Exception as e:
            print("Error al obtener el modo de trigger:", e)
            self.close()

    def set_trigger_mode(self, mode):
        modes = ('Auto', 'Normal')
        try:
            self.dso.write(f':trigger:mode {mode+1}')
            current_mode = self.get_trigger_mode()
            if current_mode == modes[mode]:
                print(f'Modo de trigger configurado a: {current_mode}')
            else:
                print('No se pudo configurar el modo de trigger.')
        except Exception as e:
            print("Error al configurar el modo de trigger:", e)
            self.close()

    def get_trigger_nrej(self):
        states = ('OFF', 'ON')
        try:
            nrej = self.dso.query(':trigger:nrej?')
            status = states[int(nrej)]
            print(f'Rechazo de ruido de trigger está {status}')
            return status
        except Exception as e:
            print("Error al obtener el estado de rechazo de ruido de trigger:", e)
            self.close()

    def set_trigger_nrej(self, state):
        states = ('OFF', 'ON')
        try:
            self.dso.write(f':trigger:nrej {state}')
            current_state = self.get_trigger_nrej()
            if current_state == states[int(state)]:
                print(f'Rechazo de ruido de trigger configurado a: {current_state}')
            else:
                print('No se pudo configurar el rechazo de ruido de trigger.')
        except Exception as e:
            print("Error al configurar el rechazo de ruido de trigger:", e)
            self.close()

    def get_trigger_reject(self):
        modes = ('OFF', 'LF', 'HF')
        try:
            rej = self.dso.query(':trigger:reject?')
            print(f'Filtro de ruido de trigger: {modes[int(rej)]}')
            return modes[int(rej)]
        except Exception as e:
            print("Error al obtener el filtro de ruido de trigger:", e)
            self.close()

    def set_trigger_reject(self, mode):
        modes = ('OFF', 'LF', 'HF')
        try:
            self.dso.write(f':trigger:reject {mode}')
            current_mode = self.get_trigger_reject()
            if current_mode == modes[mode]:
                print(f'Filtro de ruido de trigger configurado a: {current_mode}')
            else:
                print('No se pudo configurar el filtro de ruido de trigger.')
        except Exception as e:
            print("Error al configurar el filtro de ruido de trigger:", e)
            self.close()

    def get_trigger_slope(self):
        slopes = ('Positivo', 'Negativo')
        try:
            slope = self.dso.query(':trigger:slope?')
            print(f'Flanco de trigger: {slopes[int(slope)]}')
            return slopes[int(slope)]
        except Exception as e:
            print("Error al obtener el flanco de trigger:", e)
            self.close()

    def set_trigger_slope(self, slope):
        slopes = ('Positivo', 'Negativo')
        try:
            self.dso.write(f':trigger:slope {slope}')
            current_slope = self.get_trigger_slope()
            if current_slope == slopes[slope]:
                print(f'Flanco de trigger configurado a: {current_slope}')
            else:
                print('No se pudo configurar el flanco de trigger.')
        except Exception as e:
            print("Error al configurar el flanco de trigger:", e)
            self.close()

    def get_trigger_state(self):
        states = ('No disparado', 'Disparado')
        try:
            state = self.dso.query(':trigger:state?')
            print(f'Estado de trigger: {states[int(state)]}')
            return states[int(state)]
        except Exception as e:
            print("Error al obtener el estado de trigger:", e)
            self.close()

    def get_trigger_source(self):
        sources = ('Canal 1', 'Canal 2', 'Externo', 'Red')
        try:
            source = self.dso.query(':trigger:source?')
            print(f'Fuente de trigger: {sources[int(source)]}')
            return sources[int(source)]
        except Exception as e:
            print("Error al obtener la fuente de trigger:", e)
            self.close()

    def set_trigger_source(self, source):
        sources = ('Canal 1', 'Canal 2', 'Externo', 'Red')
        try:
            self.dso.write(f':trigger:source {source}')
            current_source = self.get_trigger_source()
            if current_source == sources[source]:
                print(f'Fuente de trigger configurada a: {current_source}')
            else:
                print('No se pudo configurar la fuente de trigger.')
        except Exception as e:
            print("Error al configurar la fuente de trigger:", e)
            self.close()

    def get_trigger_type(self):
        types = ('Edge', 'Video', 'Pulse')
        try:
            ttype = self.dso.query(':trigger:type?')
            print(f'Tipo de trigger: {types[int(ttype)]}')
            return types[int(ttype)]
        except Exception as e:
            print("Error al obtener el tipo de trigger:", e)
            self.close()

    def set_trigger_type(self, ttype):
        types = ('Edge', 'Video', 'Pulse')
        try:
            self.dso.write(f':trigger:type {ttype}')
            current_type = self.get_trigger_type()
            if current_type == types[ttype]:
                print(f'Tipo de trigger configurado a: {current_type}')
            else:
                print('No se pudo configurar el tipo de trigger.')
        except Exception as e:
            print("Error al configurar el tipo de trigger:", e)
            self.close()

    def get_acquire_mode(self):
        modes = ('Normal', 'Peak detect', 'Average')
        try:
            mode = self.dso.query(':acquire:mode?')
            print(f'Modo de adquisición: {modes[int(mode)]}')
            return modes[int(mode)]
        except Exception as e:
            print("Error al obtener el modo de adquisición:", e)
            self.close()

    def set_acquire_mode(self, mode):
        modes = ('Normal', 'Peak detect', 'Average')
        try:
            self.dso.write(f':acquire:mode {mode}')
            current_mode = self.get_acquire_mode()
            if current_mode == modes[mode]:
                print(f'Modo de adquisición configurado a: {current_mode}')
            else:
                print('No se pudo configurar el modo de adquisición.')
        except Exception as e:
            print("Error al configurar el modo de adquisición:", e)
            self.close()

    def get_channel_coupling(self, channel):
        couplings = ('AC', 'DC', 'GND')
        try:
            coupling = self.dso.query(f':channel{channel}:coupling?')
            print(f'Acoplamiento del canal {channel}: {couplings[int(coupling)]}')
            return couplings[int(coupling)]
        except Exception as e:
            print("Error al obtener el acoplamiento del canal:", e)
            self.close()

    def set_channel_coupling(self, channel, coupling):
        couplings = ('AC', 'DC', 'GND')
        try:
            self.dso.write(f':channel{channel}:coupling {coupling}')
            current_coupling = self.get_channel_coupling(channel)
            if current_coupling == couplings[coupling]:
                print(f'Acoplamiento del canal {channel} configurado a: {current_coupling}')
            else:
                print('No se pudo configurar el acoplamiento del canal.')
        except Exception as e:
            print("Error al configurar el acoplamiento del canal:", e)
            self.close()

    def get_channel_display(self, channel):
        states = ('OFF', 'ON')
        try:
            display = self.dso.query(f':channel{channel}:display?')
            status = states[int(display)]
            print(f'Canal {channel} está {status}')
            return status
        except Exception as e:
            print("Error al obtener el estado de visualización del canal:", e)
            self.close()

    def set_channel_display(self, channel, state):
        states = ('OFF', 'ON')
        try:
            self.dso.write(f':channel{channel}:display {state}')
            current_state = self.get_channel_display(channel)
            if current_state == states[int(state)]:
                print(f'Canal {channel} configurado a: {current_state}')
            else:
                print('No se pudo configurar el estado de visualización del canal.')
        except Exception as e:
            print("Error al configurar el estado de visualización del canal:", e)
            self.close()

    def get_channel_offset(self, channel):
        try:
            offset = float(self.dso.query(f':channel{channel}:offset?'))
            print(f'Offset del canal {channel}: {offset} [V]')
            return offset
        except Exception as e:
            print("Error al obtener el offset del canal:", e)
            self.close()

    def set_channel_offset(self, channel, offset):
        try:
            self.dso.write(f':channel{channel}:offset {offset}')
            current_offset = self.get_channel_offset(channel)
            if current_offset == offset:
                print(f'Offset del canal {channel} configurado a: {current_offset} [V]')
            else:
                print('No se pudo configurar el offset del canal.')
        except Exception as e:
            print("Error al configurar el offset del canal:", e)
            self.close()

    def get_channel_attenuation(self, channel):
        try:
            attenuation = float(self.dso.query(f':channel{channel}:probe:ratio?'))
            print(f'Factor de atenuación del canal {channel}: {attenuation}')
            return attenuation
        except Exception as e:
            print("Error al obtener el factor de atenuación del canal:", e)
            self.close()

    def set_channel_attenuation(self, channel, attenuation):
        try:
            self.dso.write(f':channel{channel}:probe:ratio {attenuation}')
            current_attenuation = self.get_channel_attenuation(channel)
            if current_attenuation == attenuation:
                print(f'Factor de atenuación del canal {channel} configurado a: {current_attenuation}')
            else:
                print('No se pudo configurar el factor de atenuación del canal.')
        except Exception as e:
            print("Error al configurar el factor de atenuación del canal:", e)
            self.close()

    def get_channel_type(self, channel):
        types = ('Tensión', 'Corriente')
        try:
            ctype = self.dso.query(f':channel{channel}:probe:type?')
            print(f'Tipo de prueba del canal {channel}: {types[int(ctype)]}')
            return types[int(ctype)]
        except Exception as e:
            print("Error al obtener el tipo de prueba del canal:", e)
            self.close()

    def set_channel_type(self, channel, ctype):
        types = ('Tensión', 'Corriente')
        try:
            self.dso.write(f':channel{channel}:probe:type {ctype}')
            current_type = self.get_channel_type(channel)
            if current_type == types[ctype]:
                print(f'Tipo de prueba del canal {channel} configurado a: {current_type}')
            else:
                print('No se pudo configurar el tipo de prueba del canal.')
        except Exception as e:
            print("Error al configurar el tipo de prueba del canal:", e)
            self.close()

    def set_single_trigger(self):
        try:
            self.dso.write(':single')
            print("Disparo único configurado exitosamente.")
        except Exception as e:
            print("Error al configurar el disparo único:", e)
            self.close()

    def close(self):
        # Cierra la conexión con el instrumento y el gestor de recursos.
        if self.dso:
            try:
                self.dso.close()
                print("Conexión con el instrumento cerrada exitosamente.")
            except Exception as e:
                print("Error al cerrar la conexión con el instrumento:", e)
            finally:
                self.dso = None
        if hasattr(self, 'rm') and self.rm is not None:
            try:
                self.rm.close()
                print("Gestor de recursos cerrado exitosamente.")
            except Exception as e:
                print("Error al cerrar el gestor de recursos:", e)

    def __del__(self):
        # Asegura que los recursos se liberen cuando el objeto es destruido.
        self.close()

    #----------------------------------------------------------------------------------------------
    @staticmethod
    def process_multipliers(value, unit):
        multipliers = {
            "V": 1.0,        # Volt.
            "mV": 1e-3,      # Milivolt.
            "uV": 1e-6,      # Microvolt.
            "s": 1.0,        # Segundo.
            "ms": 1e-3,      # Milisegundo.
            "µs": 1e-6,      # Microsegundo.
            "ns": 1e-9       # Nanosegundo.
        }
        
        try:
            numerical_value = float(value)
            factor = multipliers.get(unit, 1.0)
            scale = numerical_value * factor
            
            print(f"El parámetro final generado es: {scale}")
            return scale

        except ValueError:
            # Por si el usuario dejó el combo_valor en blanco o no es un número.
            print("Esperando un número válido en la lista...")
            return None