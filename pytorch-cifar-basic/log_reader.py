'''
read the result from the log
'''
import os
from datetime import datetime


if __name__ == "__main__":
    experiment_path = os.path.abspath('./experiment')
    model_list = os.listdir(experiment_path)
    model_list.sort()
    for model in model_list:
        if not model[0] == '.':
            print(model)
            model_log_path = os.path.join(experiment_path, model, 'logger.log')
            if os.path.isfile(model_log_path):
                with open(model_log_path) as f:
                    log_lines = f.readlines()
                start_time = datetime.strptime(log_lines[0][0:17], '%m/%d %I:%M:%S %p')
                end_time = datetime.strptime(log_lines[-1][0:17], '%m/%d %I:%M:%S %p')
                training_hours = (end_time - start_time).total_seconds() / 3600.
                print("Training times is {0}".format(training_hours))
                _temp = []
                for line in log_lines:
                    if 'parameter size' in line:
                        print(line)
                    if 'FLOPS' in line:
                        print(line)
                    if 'Current best' in line:
                        _temp.append(line)
                if len(_temp) > 0:
                    print(_temp[-1])
                else:
                    print(_temp)

