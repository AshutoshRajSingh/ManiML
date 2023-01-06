from manim import *

from optimizers import linregoptimizer
import util


class BatchGradientDescent(Scene):
    lr = 0.01
    epoch_count = 120

    epoch_animation_step = 1

    data_x_range = [-1, 1, 0.2]
    data_y_range = [-10, 10, 2]

    loss_x_buff = 10
    loss_y_range = [0, 40, 4]

    ax_scale_factor = 0.5
    info_scale_factor = 1.0

    decimal_matrix_config = {
        'bracket_h_buff': 0.5,
        'element_to_mobject_config': {
            'num_decimal_places': 3
        }
    }

    axes_config = {
        'include_numbers': True
    }
    axes_physical_side = 10.0

    op_class = linregoptimizer.BatchGradientDescentOptimizer

    def construct(self):
        x, y = self.obtain_data()

        weights, biases = self.get_weights_and_biases(x, y)
        losses, epochs = self.get_losses(x, y, weights, biases), list(range(1, self.epoch_count + 1))

        first_weight, first_bias = weights[0], biases[0]

        weight_matrix = DecimalMatrix(
            first_weight,
            **self.decimal_matrix_config
        )

        weight_matrix_label = MathTex(
            '\\theta',
            '='
        ).next_to(weight_matrix, LEFT)

        weight_matrix_group = VGroup(weight_matrix, weight_matrix_label)

        bias_matrix = DecimalMatrix(
            [first_bias],
            **self.decimal_matrix_config
        )

        bias_matrix_label = MathTex(
            'b',
            '='
        ).next_to(bias_matrix, LEFT)

        bias_matrix_group = VGroup(bias_matrix, bias_matrix_label).next_to(
            weight_matrix_group, RIGHT)

        epoch_counter = Integer(
            1
        )

        epoch_label = Tex(
            'Epoch: '
        ).next_to(epoch_counter, LEFT)

        epoch_info = VGroup(epoch_counter, epoch_label).next_to(
            bias_matrix_group, RIGHT)

        info_group = VGroup(weight_matrix_group,
                            bias_matrix_group, epoch_info).scale(self.info_scale_factor).center().to_edge(UP)

        data_ax = Axes(
            self.data_x_range,
            self.data_y_range,
            self.axes_physical_side,
            self.axes_physical_side,
            axis_config=self.axes_config
        )

        loss_ax = Axes(
            self.get_loss_x_range(),
            self.loss_y_range,
            self.axes_physical_side,
            self.axes_physical_side,
            axis_config=self.axes_config
        ).next_to(data_ax, RIGHT, buff=1)

        ax_group = VGroup(data_ax, loss_ax).scale(
            self.ax_scale_factor).center().to_edge(DOWN)

        data_points = self.plot_data_points(data_ax, x, y)

        model_line = data_ax.plot(
            lambda x: self.model_curve_fn(x, first_weight, first_bias)).set_color(LIGHT_PINK)

        loss_plot = loss_ax.plot_line_graph(
            epochs[0:1],
            losses[0:1],
            vertex_dot_radius=0.0,
            line_color=RED
        )

        self.play(
            Create(ax_group, 0.0),
            Create(loss_plot)
        )
        self.play(
            FadeIn(*data_points, loss_plot),
            Create(model_line),
            run_time=0.5
        )
        self.play(
            Write(info_group),
            run_time=1.5
        )
        self.wait()

        for (idx, (weight, bias)) in enumerate(zip(weights[1::self.epoch_animation_step], biases[1::self.epoch_animation_step])):
            self.play(
                Transform(
                    weight_matrix,
                    DecimalMatrix(
                        weight,
                        **self.decimal_matrix_config
                    ).move_to(weight_matrix)
                ),
                Transform(
                    bias_matrix,
                    DecimalMatrix(
                        [bias],
                        **self.decimal_matrix_config
                    ).move_to(bias_matrix)
                ),
                Transform(
                    epoch_counter,
                    Integer(
                        (idx + 1) * self.epoch_animation_step
                    ).next_to(epoch_label, RIGHT)
                ),
                Transform(
                    model_line,
                    data_ax.plot(
                        lambda x: self.model_curve_fn(x, weight, bias),
                    ).set_color(LIGHT_PINK)
                ),
                Transform(
                    loss_plot,
                    loss_ax.plot_line_graph(
                        epochs[:(idx + 1) * self.epoch_animation_step:self.epoch_animation_step],
                        losses[:(idx + 1) * self.epoch_animation_step:self.epoch_animation_step],
                        vertex_dot_radius=0.0,
                        line_color=RED,
                    )
                ),
                run_time=0.1
            )
        self.wait(5)

    def model_curve_fn(self, x, weight_v, bias):
        weight_num = weight_v[0, 0]
        bias_num = bias[0]

        return x * weight_num + bias_num
    
    def obtain_data(self):
        return util.generate_dummy_linear_data()

    def get_weights_and_biases(self, x, y):
        op = self.op_class(self.lr)
        return op.fit_remembering_weights(x, y, self.epoch_count)
    
    def loss_fn(self, x, y, weight_v, bias):
        return util.mse_loss(x, y, weight_v, bias)
    
    def get_losses(self, x, y, weights, biases):
        losses = []

        for weight_v, bias in zip(weights, biases):
            losses.append(self.loss_fn(x, y, weight_v, bias))
        
        return losses
    
    def plot_data_points(self, data_ax, x, y):
        return [data_ax.plot_line_graph(
            x.ravel(), y.ravel(),
            line_color=None,
            vertex_dot_radius=0.02,
            vertex_dot_style={'color': YELLOW}
        )]
    
    def get_loss_x_range(self):
        return [0, self.epoch_count + self.loss_x_buff, (self.epoch_count + self.loss_x_buff) // 10]


class StochasticGradientDescent(BatchGradientDescent):
    lr = 0.1
    op_class = linregoptimizer.StochasticGradientDescentOptimizer
