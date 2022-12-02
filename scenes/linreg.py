from optimizers.lro import BatchGradientDescentOptimizer
from manim import *
import numpy as np

import sys
import os
sys.path.append(os.curdir)


def generate_dummy_linear_data(n=100, w=3, b=4):
    x = np.random.rand(n, 1)
    y = b + w * x + np.random.rand(n, 1)

    return x, y


DEFAULT_DECIMAL_MATRIX_CONFIG = {
    'bracket_h_buff': 0.5,
    'element_to_mobject_config': {
        'num_decimal_places': 3
    }
}


class BatchGradientDescent(Scene):
    op_class = BatchGradientDescentOptimizer

    def construct(self):
        x, y = generate_dummy_linear_data()
        weights, biases = self.get_weights_and_biases(x, y)

        first_weight, first_bias = weights[0], biases[0]

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

        epoch_info = Tex(
            'Epoch: ',
            '10000'
        ).next_to(bias_matrix_group, RIGHT)

        info_group = VGroup(weight_matrix_group,
                            bias_matrix_group, epoch_info).center().to_edge(UP)

    def get_weights_and_biases(self, x, y, epochs=1000, lr=0.01):
        op = self.op_class(lr)
        return op.fit_remembering_weights(x, y, epochs)
