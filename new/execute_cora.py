import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import GAT
from process import load_data, preprocess_features

# 데이터 로드
dataset = "cora"
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(
    dataset
)
features = preprocess_features(features)

features = torch.FloatTensor(features)
adj = torch.FloatTensor(adj.to_dense())
y_train = torch.LongTensor(y_train)
y_val = torch.LongTensor(y_val)
y_test = torch.LongTensor(y_test)
train_mask = torch.BoolTensor(train_mask)
val_mask = torch.BoolTensor(val_mask)
test_mask = torch.BoolTensor(test_mask)

# 모델 초기화
model = GAT(
    n_feat=features.shape[1],
    n_hid=8,
    n_class=y_train.shape[1],
    dropout=0.6,
    n_heads=8,
)
optimizer = Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


# 학습 함수 정의
def train(epoch):
    # 학습 상태로 설정
    model.train()
    # optimizer gradient 초기화
    optimizer.zero_grad()
    # 학습
    output = model(features, adj)
    # loss, 정확도 계산
    loss = F.cross_entropy(output[train_mask], y_train.argmax(dim=1))
    acc = (output[train_mask].argmax(dim=1) == y_train.argmax(dim=1)).float().mean()
    # backpropagation, 모델 파라미터 계산
    loss.backward()
    # 모델 파라미터 업데이트
    optimizer.step()

    print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")


# 평가 함수 정의
def evaluate():
    # 검증 상태로 설정
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = F.cross_entropy(output[val_mask], y_val.argmax(dim=1))
        acc = (output[val_mask].argmax(dim=1) == y_val.argmax(dim=1)).float().mean()
    return loss.item(), acc.item()


# 훈련 루프 (early stopping 간략 구현)
best_acc = 0
patience = 10
wait = 0

# best_acc를 연속 10회 이상 갱신하지 못하면 early stop!
for epoch in range(1, 100):
    train(epoch)
    val_loss, val_acc = evaluate()
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        wait = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        wait += 1
        if wait == patience:
            print("Early stopping!")
            break

# 테스트 데이터 평가
model.load_state_dict(torch.load("best_model.pth"))
model.eval()
with torch.no_grad():
    output = model(features, adj)
    loss_test = F.cross_entropy(output[test_mask], y_test.argmax(dim=1))
    acc_test = (output[test_mask].argmax(dim=1) == y_test.argmax(dim=1)).float().mean()

print(f"Test Loss: {loss_test.item():.4f}, Test Accuracy: {acc_test.item():.4f}")
