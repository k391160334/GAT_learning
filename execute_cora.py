import time
import numpy as np
import tensorflow as tf

from models import GAT
from utils import process

checkpt_file = "pre_trained/cora/mod_cora.ckpt"

dataset = "cora"

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8]  # numbers of hidden units per each attention head in each layer
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print("Dataset: " + dataset)
print("----- Opt. hyperparams -----")
print("lr: " + str(lr))
print("l2_coef: " + str(l2_coef))
print("----- Archi. hyperparams -----")
print("nb. layers: " + str(len(hid_units)))
print("nb. units per layer: " + str(hid_units))
print("nb. attention heads: " + str(n_heads))
print("residual: " + str(residual))
print("nonlinearity: " + str(nonlinearity))
print("model: " + str(model))

adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = (
    process.load_data(dataset)
)
# 특징 행렬 정규화
features, spars = process.preprocess_features(features)

nb_nodes = features.shape[0]  # 노드 개수
ft_size = features.shape[1]  # 특징 개수
nb_classes = y_train.shape[1]  # 클래스 개수

# 인접행렬(adj)을 밀집행렬로 변환 (Sparse → Dense)
adj = adj.todense()

# 차원을 확장하여 배치 차원을 추가
features = features[np.newaxis]  # [batch_size, nb_nodes, ft_size]
adj = adj[np.newaxis]  # [batch_size, nb_nodes, nb_nodes]
y_train = y_train[np.newaxis]  # [batch_size, nb_nodes, nb_classes]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

# 연결된 노드 사이에서만 attention이 계산되도록 할 bias 행렬 생성, [batch_size, nb_nodes, nb_nodes]
# 그래프에서, 연결되지 않은 부분을 -무한대 값으로, 연결된 부분은 0으로 변환
biases = process.adj_to_bias(adj, [nb_nodes], nhood=1)

with tf.Graph().as_default():
    # 1. Tensorflow 모델 그래프 구축 및 훈련 루프
    with tf.name_scope("input"):
        ftr_in = tf.placeholder(
            dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size)
        )  # 특징
        bias_in = tf.placeholder(  # 그래프 연결 정보
            dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes)
        )
        lbl_in = tf.placeholder(  # 노드 레이블
            dtype=tf.int32, shape=(batch_size, nb_nodes, nb_classes)
        )
        msk_in = tf.placeholder(
            dtype=tf.int32, shape=(batch_size, nb_nodes)
        )  # 마스크 (훈련, 검증, 테스트 노드 구분)
        attn_drop = tf.placeholder(dtype=tf.float32, shape=())  # Attention dropout 비율
        ffd_drop = tf.placeholder(
            dtype=tf.float32, shape=()
        )  # feed forward dropout 비율
        is_train = tf.placeholder(dtype=tf.bool, shape=())  # 훈련 여부 플래그

    # 모델 inference (예측값 계산)
    logits = model.inference(
        ftr_in,
        nb_classes,
        nb_nodes,
        is_train,
        attn_drop,
        ffd_drop,
        bias_mat=bias_in,
        hid_units=hid_units,
        n_heads=n_heads,
        residual=residual,
        activation=nonlinearity,
    )
    # logits 형태를 평가하기 쉽게 2차원으로 변형, 손실과 정확도 계산을 위한 레이블과 마스크도 reshape
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    # 손실, 정확도 계산
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)

    # 최적화 연산 정의 (손실 최소화)
    train_op = model.training(loss, lr, l2_coef)

    # 모델 저장 및 초기화 연산 정의
    saver = tf.train.Saver()
    init_op = tf.group(
        tf.global_variables_initializer(), tf.local_variables_initializer()
    )

    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0

    with tf.Session() as sess:
        # 2. 학습 루프 및 early stopping
        sess.run(init_op)  # 변수 초기화

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0

        for epoch in range(nb_epochs):
            # 훈련
            tr_step = 0
            tr_size = features.shape[0]  # 그래프 개수

            while tr_step * batch_size < tr_size:
                _, loss_value_tr, acc_tr = (
                    sess.run(  # 실제로 model inference 코드 실행하고, backpropagation까지 일어남.
                        [train_op, loss, accuracy],
                        feed_dict={
                            ftr_in: features[
                                tr_step * batch_size : (tr_step + 1) * batch_size
                            ],
                            bias_in: biases[
                                tr_step * batch_size : (tr_step + 1) * batch_size
                            ],
                            lbl_in: y_train[
                                tr_step * batch_size : (tr_step + 1) * batch_size
                            ],
                            msk_in: train_mask[
                                tr_step * batch_size : (tr_step + 1) * batch_size
                            ],
                            is_train: True,
                            attn_drop: 0.6,
                            ffd_drop: 0.6,
                        },
                    )
                )
                train_loss_avg += loss_value_tr
                train_acc_avg += acc_tr
                tr_step += 1

            # 검증
            vl_step = 0
            vl_size = features.shape[0]

            while vl_step * batch_size < vl_size:
                loss_value_vl, acc_vl = sess.run(
                    [loss, accuracy],
                    feed_dict={
                        ftr_in: features[
                            vl_step * batch_size : (vl_step + 1) * batch_size
                        ],
                        bias_in: biases[
                            vl_step * batch_size : (vl_step + 1) * batch_size
                        ],
                        lbl_in: y_val[
                            vl_step * batch_size : (vl_step + 1) * batch_size
                        ],
                        msk_in: val_mask[
                            vl_step * batch_size : (vl_step + 1) * batch_size
                        ],
                        is_train: False,
                        attn_drop: 0.0,
                        ffd_drop: 0.0,
                    },
                )
                val_loss_avg += loss_value_vl
                val_acc_avg += acc_vl
                vl_step += 1

            print(
                "Training: loss = %.5f, acc = %.5f | Val: loss = %.5f, acc = %.5f"
                % (
                    train_loss_avg / tr_step,
                    train_acc_avg / tr_step,
                    val_loss_avg / vl_step,
                    val_acc_avg / vl_step,
                )
            )

            # 검증 성능을 기반으로 조기 종료 조건 체크
            # 조건에 만족할 경우 훈련 종료
            if val_acc_avg / vl_step >= vacc_mx or val_loss_avg / vl_step <= vlss_mn:
                if (
                    val_acc_avg / vl_step >= vacc_mx
                    and val_loss_avg / vl_step <= vlss_mn
                ):
                    vacc_early_model = val_acc_avg / vl_step
                    vlss_early_model = val_loss_avg / vl_step
                    saver.save(sess, checkpt_file)
                vacc_mx = np.max((val_acc_avg / vl_step, vacc_mx))
                vlss_mn = np.min((val_loss_avg / vl_step, vlss_mn))
                curr_step = 0
            else:
                curr_step += 1
                if curr_step == patience:
                    print(
                        "Early stop! Min loss: ", vlss_mn, ", Max accuracy: ", vacc_mx
                    )
                    print(
                        "Early stop model validation loss: ",
                        vlss_early_model,
                        ", accuracy: ",
                        vacc_early_model,
                    )
                    break

            train_loss_avg = 0
            train_acc_avg = 0
            val_loss_avg = 0
            val_acc_avg = 0

        # 3. 테스트 데이터 평가
        saver.restore(sess, checkpt_file)

        ts_size = features.shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            loss_value_ts, acc_ts = sess.run(
                [loss, accuracy],
                feed_dict={
                    ftr_in: features[ts_step * batch_size : (ts_step + 1) * batch_size],
                    bias_in: biases[ts_step * batch_size : (ts_step + 1) * batch_size],
                    lbl_in: y_test[ts_step * batch_size : (ts_step + 1) * batch_size],
                    msk_in: test_mask[
                        ts_step * batch_size : (ts_step + 1) * batch_size
                    ],
                    is_train: False,
                    attn_drop: 0.0,
                    ffd_drop: 0.0,
                },
            )
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print("Test loss:", ts_loss / ts_step, "; Test accuracy:", ts_acc / ts_step)

        sess.close()
