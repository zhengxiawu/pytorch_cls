from pycls.core.plotting import plot_error_curves_plotly, plot_error_curves_pyplot
import argparse

if __name__ == "__main__":
    log_file_name = ['./experiments/EN-B0_dds_4gpu/stdout.log']
    name = ['efficientB0']
    filename = './experiments/EN-B0_dds_4gpu/result.html'
    plot_error_curves_plotly(log_file_name, name, filename)
    pass
