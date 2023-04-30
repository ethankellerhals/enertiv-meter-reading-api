import subprocess

def main():

    try:
        detect_meter_model = subprocess.Popen(['python', 'model.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return_meter_type = detect_meter_model.wait()
        meter_type, error = detect_meter_model.communicate()

        if len(error) > 0:
            print(error.decode('utf-8'))
        else:
            match meter_type.decode('utf-8'):
                case "analog":
                    analog_meter_alorithm = subprocess.Popen(['python', 'analog_meter.py', 'arg1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return_analog_meter = analog_meter_alorithm.wait()
                    analog_meter_value, error = analog_meter_alorithm.communicate()

                    if len(error) > 0:
                        print(error.decode('utf-8'))
                    else:
                        print(f"analog meter value: {analog_meter_value.decode('utf-8')} kWh")
                case "dial":          
                    dial_meter_algorithm = subprocess.Popen(['python', 'dial_meters.py', 'arg1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return_dial_meter = dial_meter_algorithm.wait()
                    dial_meter_value, error = dial_meter_algorithm.communicate()

                    if len(error) > 0:
                        print(error.decode('utf-8'))
                    else:
                        print(f"dial meter value: {dial_meter_value.decode('utf-8')} kWh")
                case "digital":            
                    digital_meter_algorithm = subprocess.Popen(['python', 'digital_meter.py', 'arg1'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    return_digital_meter = digital_meter_algorithm.wait()
                    digital_meter_value, error = digital_meter_algorithm.communicate()

                    if len(error) > 0:
                        print(error.decode('utf-8'))
                    else:
                        print(f"digital meter value: {digital_meter_value.decode('utf-8')} kWh")
                case _:
                    print("Error: could not detect a energy meter")
    except Exception as e:
        print(f"An error occured: {e}")

if __name__ == '__main__':
    main()