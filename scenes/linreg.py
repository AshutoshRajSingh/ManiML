from manim import *
import numpy as np

import sys
import os
sys.path.append(os.curdir)

from optimizers.lro import BatchGradientDescentOptimizer


def generate_dummy_linear_data(n=100, w=3, b=4):
    x = np.random.rand(n, 1)
    y = b + w * x + np.random.rand(n, 1)

    return x, y


def mse_loss(x, y, weights, bias):
    return np.mean((x.dot(weights) + bias - y) ** 2)


DEFAULT_DECIMAL_MATRIX_CONFIG = {
    'bracket_h_buff': 0.5,
    'element_to_mobject_config': {
        'num_decimal_places': 3
    }
}

DEFAULT_AX_CONFIG = {
    'include_numbers': True
}

AX_DEFAULT_SCALE_FACTOR = 0.5
MAX_EPOCH = 120 + 1
LR_DEFAULT = 0.01
AX_SIDE = 10
DATA_X_RANGE_DEFAULT = [-1, 1, 0.2]
DATA_Y_RANGE_DEFAULT = [-10, 10, 2]
LOSS_X_RANGE_DEFAULT = [0, MAX_EPOCH, MAX_EPOCH // 10]
LOSS_Y_RANGE_DEFAULT = [0, 40, 4]


class BatchGradientDescent(Scene):
    op_class = BatchGradientDescentOptimizer

    def construct(self):
        x, y = generate_dummy_linear_data()
        weights, biases = self.get_weights_and_biases(x, y)

        losses, epochs = [0], [0]

        first_weight, first_bias = weights[0], biases[0]

        losses[0] = mse_loss(
            x, y, first_weight[:, np.newaxis], first_bias[:, np.newaxis])

        weight_matrix = DecimalMatrix(
            [first_weight],
            **DEFAULT_DECIMAL_MATRIX_CONFIG
        )

        weight_matrix_label = MathTex(
            '\\theta',
            '='
        ).next_to(weight_matrix, LEFT)

        weight_matrix_group = VGroup(weight_matrix, weight_matrix_label)

        bias_matrix = DecimalMatrix(
            [first_bias],
            **DEFAULT_DECIMAL_MATRIX_CONFIG
        )

        bias_matrix_label = MathTex(
            'b',
            '='
        ).next_to(bias_matrix, LEFT)

        bias_matrix_group = VGroup(bias_matrix, bias_matrix_label).next_to(
            weight_matrix_group, RIGHT)

        epoch_counter = Integer(
            0
        )

        epoch_label = Tex(
            'Epoch: '
        ).next_to(epoch_counter, LEFT)

        epoch_info = VGroup(epoch_counter, epoch_label).next_to(
            bias_matrix_group, RIGHT)

        info_group = VGroup(weight_matrix_group,
                            bias_matrix_group, epoch_info).center().to_edge(UP)

        data_ax = Axes(
            DATA_X_RANGE_DEFAULT,
            DATA_Y_RANGE_DEFAULT,
            AX_SIDE,
            AX_SIDE,
            axis_config=DEFAULT_AX_CONFIG
        )

        loss_ax = Axes(
            LOSS_X_RANGE_DEFAULT,
            LOSS_Y_RANGE_DEFAULT,
            AX_SIDE,
            AX_SIDE,
            axis_config=DEFAULT_AX_CONFIG
        ).next_to(data_ax, RIGHT, buff=1)

        ax_group = VGroup(data_ax, loss_ax).scale(
            AX_DEFAULT_SCALE_FACTOR).center().to_edge(DOWN)

        data_points = data_ax.plot_line_graph(
            x.ravel(), y.ravel(),
            line_color=None,
            vertex_dot_radius=0.02,
            vertex_dot_style={'color': YELLOW}
        )

        model_line = data_ax.plot(
            lambda x: x * first_weight[0] + first_bias[0]).set_color(LIGHT_PINK)

        loss_plot = loss_ax.plot_line_graph(
            epochs,
            losses,
            vertex_dot_radius=0.0,
            line_color=RED
        )

        self.play(
            Create(ax_group, 0.0),
            Create(loss_plot)
        )
        self.play(
            FadeIn(data_points, loss_plot),
            run_time=0.5
        )
        self.play(
            Create(model_line),
            run_time=0.5
        )
        self.play(
            Write(info_group),
            run_time=1.5
        )
        self.wait()

        for (idx, (weight, bias)) in enumerate(zip(weights[1::], biases[1::])):
            losses.append(
                mse_loss(x, y, weight[:, np.newaxis], bias[:, np.newaxis]))
            epochs.append(idx + 1)

            self.play(
                Transform(
                    weight_matrix,
                    DecimalMatrix(
                        [weight],
                        **DEFAULT_DECIMAL_MATRIX_CONFIG
                    ).move_to(weight_matrix)
                ),
                Transform(
                    bias_matrix,
                    DecimalMatrix(
                        [bias],
                        **DEFAULT_DECIMAL_MATRIX_CONFIG
                    ).move_to(bias_matrix)
                ),
                Transform(
                    epoch_counter,
                    Integer(
                        idx + 1
                    ).next_to(epoch_label, RIGHT)
                ),
                Transform(
                    model_line,
                    data_ax.plot(
                        lambda x: weight[0] * x + bias[0]
                    ).set_color(LIGHT_PINK)
                ),
                Transform(
                    loss_plot,
                    loss_ax.plot_line_graph(
                        epochs,
                        losses,
                        vertex_dot_radius=0.0,
                        line_color=RED
                    )
                ),
                run_time=0.1
            )
        self.wait(5)

    def get_weights_and_biases(self, x, y, epochs=MAX_EPOCH, lr=LR_DEFAULT):
        op = self.op_class(lr)
        self.op = op
        return op.fit_remembering_weights(x, y, epochs)
