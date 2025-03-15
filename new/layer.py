import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha  # LeakyReLU의 음수 기울기
        self.concat = concat

        self.W = nn.Parameter(  # weight matrix
            torch.empty(size=(in_features, out_features))
        )
        self.a = nn.Parameter(  # attention 메커니즘의 가중치, concat된 두 노드 특징 벡터를 하나의 attention 점수(스칼라)로 변환
            torch.empty(size=(2 * out_features, 1))
        )

        self.leakyrelu = nn.LeakyReLU(self.alpha)  # activation function
        self.reset_parameters()  # model parameter 초기화

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # [nb_node, out_features]
        a_input = self._prepare_attentional_mechanism_input(
            Wh
        )  # [nb_node, nb_node, out_features * 2]
        e = torch.matmul(a_input, self.a).squeeze(2)  # [nb_node, nb_node]
        e = self.leakyrelu(e)

        # 연결된 노드만 attention 점수 부여(마스킹 적용 효과)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        # 정규화
        attention = F.softmax(attention, dim=1)
        # dropout
        attention = F.dropout(attention, self.dropout, training=self.training)
        # attention 기반의 노드 특징 업데이트, [nb_nodes, out_features]
        h_prime = torch.matmul(attention, Wh)

        if (
            self.concat
        ):  # 중간의 attention layer일 때 ELU 활성화 함수를 적용하여 비선형성을 추가
            return F.elu(h_prime)
        else:  # 마지막 layer
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # 브로드 캐스팅
        """
        Wh1[0] = [Wh[0], Wh[0], Wh[0], ..., Wh[0]]
        Wh1[1] = [Wh[1], Wh[1], Wh[1], ..., Wh[1]]
        ...
        Wh1[N-1] = [Wh[N-1], Wh[N-1], Wh[N-1], ..., Wh[N-1]]
        --------------
        Wh2[0] = [Wh[0], Wh[1], Wh[2], ..., Wh[N-1]]
        Wh2[1] = [Wh[0], Wh[1], Wh[2], ..., Wh[N-1]]
        ...
        Wh2[N-1] = [Wh[0], Wh[1], Wh[2], ..., Wh[N-1]]
        """
        Wh1 = Wh.unsqueeze(1).repeat(
            1, Wh.size(0), 1
        )  # [nb_node, nb_node, out_features]
        Wh2 = Wh.unsqueeze(0).repeat(
            Wh.size(0), 1, 1
        )  # [nb_node, nb_node, out_features]
        # 더해줌
        return torch.cat(
            [Wh1, Wh2], dim=2
        )  # [nb_node, nb_node, out_features*2] , 반환값[i][j] = [Wh[i] || Wh[j]]
