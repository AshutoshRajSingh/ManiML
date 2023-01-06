import util
from scenes import linreg
from optimizers import logregoptimizer

class BatchGradientDescent(linreg.BatchGradientDescent):
    lr = 0.01
    op_class = logregoptimizer.BatchGradientDescentOptimizer
    data_y_range = [-4, 4, 1.5]
    data_x_range = [-7, 7, 2]

    loss_y_range = [0, 2, 0.1]
    epoch_count = 10000

    axes_physical_side = 8

    epoch_animation_step = 1000

    def model_curve_fn(self, x, weight_v, bias):
        return -(x * weight_v[0, 0] + bias[0]) / weight_v[1, 0]

    def obtain_data(self):
        return util.obtain_classification_data()
    
    def loss_fn(self, x, y, weight_v, bias):
        return util.log_loss(x, y, weight_v, bias)
    
    def plot_data_points(self, data_ax, x, y):
        plot1 = data_ax.plot_line_graph(
            x[:, 0][y.ravel() == 0],
            x[:, 1][y.ravel() == 0],
            line_color=None,
            vertex_dot_radius=0.02,
            vertex_dot_style={'color': '#FFFF00'}
        )
        plot2 = data_ax.plot_line_graph(
            x[:, 0][y.ravel() == 1],
            x[:, 1][y.ravel() == 1],
            line_color=None,
            vertex_dot_radius=0.02,
            vertex_dot_style={'color': '#58C4DD'}
        )
        return [plot1, plot2]

class StochasticGradientDescent(BatchGradientDescent):
    lr = 0.01
    epoch_count = 100
    op_class = logregoptimizer.StochasticGradientDescentOptimizer
