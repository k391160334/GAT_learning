import torch
import torch.nn as nn
import torch.nn.functional as F
from layer import GraphAttentionLayer


class GAT(nn.Module):
    """
    n_feat: 입력 노드의 특징 수
    n_hid: 각 Attention Head의 출력 특징 수 (hidden features)
    n_class: 최종 분류할 클래스 수
    dropout: 오버피팅 방지를 위한 dropout 비율
    n_heads: 다중 attention head의 개수
    alpha: LeakyReLU의 음수 기울기(slope), 보통 0.2 사용
    """

    def __init__(self, n_feat, n_hid, n_class, dropout, n_heads, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout

        # 첫 번째 attention layer (다중 헤드->여러 헤드를 통해 서로 다른 관점에서 attention을 계산), [nb_node, n_hid * n_heads]
        self.attentions = nn.ModuleList(
            [
                GraphAttentionLayer(n_feat, n_hid, dropout, alpha, concat=True)
                for _ in range(n_heads)
            ]
        )

        # 출력층 attention layer (단일 헤드), [nb_node, n_class]
        self.out_att = GraphAttentionLayer(
            n_hid * n_heads, n_class, dropout, alpha, concat=False
        )

    def forward(self, x, adj):
        x = F.dropout(
            x, self.dropout, training=self.training
        )  # 입력 특징에 dropout 적용
        x = torch.cat(
            [att(x, adj) for att in self.attentions], dim=1
        )  # [nb_nodes, n_heads * n_hid]
        x = F.dropout(
            x, self.dropout, training=self.training
        )  # 다중 헤드 결과에 다시 dropout 적용 (오버피팅 방지)
        x = self.out_att(x, adj)  # [nb_nodes, n_class]
        return x
