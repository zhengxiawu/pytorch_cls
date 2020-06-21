from pycls.core.plotting import plot_error_curves_plotly, plot_error_curves_pyplot
import argparse

if __name__ == "__main__":
    log_file_name = ['./experiments/mobilenetv2_dds_2gpu/stdout.log']
    name = ['MBV2']
    filename = './experiments/mobilenetv2_dds_2gpu/result.png'
    plot_error_curves_plotly(log_file_name, name, filename)
    pass
