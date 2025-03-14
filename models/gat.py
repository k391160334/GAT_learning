import numpy as np
import tensorflow as tf

from utils import layers
from models.base_gattn import BaseGAttN


class GAT(BaseGAttN):
    def inference(
        inputs,  # 특징 행렬, [batch_size, nb_nodes, feature_size]
        nb_classes,  # 클래스 개수
        nb_nodes,  # 노드 개수
        training,  # 훈련 여부
        attn_drop,  # attention dropout
        ffd_drop,  # feed forward dropout
        bias_mat,  # 연결 관계를 반영한 bias 행렬, [batch_size, nb_nodes, nb_nodes]
        hid_units,  # 각 레이어의 hidden unit(output feature) 크기
        n_heads,  # 각 레이어의 head 수
        activation=tf.nn.elu,
        residual=False,
    ):
        attns = []
        for _ in range(n_heads[0]):  # 첫 레이어
            attns.append(
                layers.attn_head(
                    inputs,
                    bias_mat=bias_mat,
                    out_sz=hid_units[0],
                    activation=activation,
                    in_drop=ffd_drop,
                    coef_drop=attn_drop,
                    residual=False,
                )
            )
        h_1 = tf.concat(attns, axis=-1)
        for i in range(1, len(hid_units)):
            h_old = h_1
            attns = []
            for _ in range(n_heads[i]):  # 중간 레이어들
                attns.append(
                    layers.attn_head(
                        h_1,
                        bias_mat=bias_mat,
                        out_sz=hid_units[i],
                        activation=activation,
                        in_drop=ffd_drop,
                        coef_drop=attn_drop,
                        residual=residual,  # 잔차 추가
                    )
                )
            h_1 = tf.concat(attns, axis=-1)
        out = []
        for i in range(n_heads[-1]):  # 마지막 레이어
            out.append(
                layers.attn_head(
                    h_1,
                    bias_mat=bias_mat,
                    out_sz=nb_classes,  # 클래스 개수
                    activation=lambda x: x,  # 별도로 activation 함수 x
                    in_drop=ffd_drop,
                    coef_drop=attn_drop,
                    residual=False,
                )
            )
        logits = tf.add_n(out) / n_heads[-1]

        return logits
