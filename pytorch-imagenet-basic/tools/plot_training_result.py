from pycls.core.plotting import plot_error_curves_plotly, plot_error_curves_pyplot
import argparse

if __name__ == "__main__":
    log_file_name = ['./experiments/EN-B0_dds_4gpu/stdout.log',
                     './experiments/EN-B0_dds_4gpu_color_PCA/stdout.log', ]
    name = ['efficientB0', 'efficientB0_PCA']
    filename = './experiments/EN-B0_dds_4gpu/result_.html'
    plot_error_curves_plotly(log_file_name, name, filename)
    pass
