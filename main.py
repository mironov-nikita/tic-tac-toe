import torch
from torch import nn
import os
import random

class TicTacToeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 18),
            nn.ReLU(),
            nn.Linear(18, 18),
            nn.ReLU(),
            nn.Linear(18, 9),
        )

    def forward(self, s):
        return self.net(s)

model = TicTacToeModel()

path = "checkpoints/tic_tac_toe.pt"
if os.path.exists(path):
    model.load_state_dict(torch.load(path))
    model.eval()
    print("The weights are loaded.")
else:
    print("No weights found.")

loss_fn = nn.L1Loss()
optimizer_fn = torch.optim.Adam(model.parameters(), lr=0.001)


def WhoHasWon(s):   
    # First horizonta
    if sum(s[:3]) == 3:
        return "Tic"
    elif sum(s[:3]) == -3:
        return "Toe"
     
    # Second horizontal
    elif sum(s[3:6]) == 3:
        return "Tic"
    elif sum(s[3:6]) == -3:
        return "Toe"
    
    # Third horizontal
    elif sum(s[6:9]) == 3:
        return "Tic"
    elif sum(s[6:9]) == -3:
        return "Toe"
    
    # First vertical
    elif s[0] + s[3] + s[6] == 3:
        return "Tic"
    elif s[0] + s[3] + s[6] == -3:
        return "Toe"
    
    # Second vertical
    elif s[1] + s[4] + s[7] == 3:
        return "Tic"
    elif s[1] + s[4] + s[7] == -3:
        return "Toe"
    
    # Third vertical
    elif s[2] + s[5] + s[8] == 3:
        return "Tic"
    elif s[2] + s[5] + s[8] == -3:
        return "Toe"
    
    # Diagonal that starts from the top left
    elif s[0] + s[4] + s[8] == 3:
        return "Tic"
    elif s[0] + s[4] + s[8] == -3:
        return "Toe"
    
    # Diagonal that starts from the bottom left 
    elif s[6] + s[4] + s[2] == 3:
        return "Tic"
    elif s[6] + s[4] + s[2] == -3:
        return "Toe"
    
    return False


def TicTacToeGame(model, epsilon=0.2):
    r = None
    s = torch.zeros(9, dtype=torch.float32) 
    MovesHistory = []
    
    for i in range(9):
        current_state = s.clone().unsqueeze(0)
        logits = model(current_state)[0]  
        mask = (s == 0)
        free_indices = mask.nonzero(as_tuple=True)[0]
        if len(free_indices) == 0:
            break

        if random.random() < epsilon:
            move_index = free_indices[random.randint(0, len(free_indices) - 1)]
            move_value = logits[move_index]
        else:
            logits_masked = logits.clone()
            logits_masked[~mask] = -float('inf')
            move_index = torch.argmax(logits_masked)
            move_value = logits_masked[move_index]
        
        MovesHistory.append((move_index, move_value))

        if i % 2 == 0:
            s[move_index] = 1  # Tic
        else:
            s[move_index] = -1 # Toe

        Winner = WhoHasWon(s)
        if Winner:
            r = 1 if Winner == "Tic" else -1
            break
            
    return r, MovesHistory

def MyLoss(MovesHistoryTic, MovesHistoryToe, r, gamma=0.99):
    loss = torch.tensor(0.0, requires_grad=True)
    target_tic = r if r is not None else 0
    for i in reversed(range(len(MovesHistoryTic))):
        move_index, move_value = MovesHistoryTic[i]
        
        loss = loss + (move_value - target_tic)**2
        target_tic *= gamma

    target_toe = -r if r is not None else 0
    for i in reversed(range(len(MovesHistoryToe))):
        move_index, move_value = MovesHistoryToe[i]
        loss = loss + (move_value - target_toe)**2
        target_toe *= gamma

    return loss / 9



def train(model, epochs=100):
    model.train()
    print("========== TRAINING STARTS ==========")
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(100):
            optimizer_fn.zero_grad()

            r, MovesHistory = TicTacToeGame(model)
            MovesHistoryTic = MovesHistory[0::2] # even
            MovesHistoryToe = MovesHistory[1::2] # odd

            Loss = MyLoss(MovesHistoryTic, MovesHistoryToe, r, 0.9)
            Loss.backward()
            optimizer_fn.step()
            
            total_loss += Loss.item()
            
        print(f"Epoch {epoch} | Loss: {total_loss/100:.4f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/tic_tac_toe.pt")
    

def TestTicTacToeModel(model):
    s = torch.zeros(9, dtype=torch.float32)
    model.eval()
    
    with torch.inference_mode():
        for turn in range(9):
            print(s.view(3, 3))
            
            if turn % 2 == 0:
                logits = model(s.unsqueeze(0))[0]
                mask = (s == 0)
                logits[~mask] = -float('inf')
                move = torch.argmax(logits).item()
                s[move] = 1
                print(f"Model move on {move + 1}")
            else:
                move = int(input("Your turn (1-9): ")) - 1
                s[move] = -1
            
            winner = WhoHasWon(s)
            if winner:
                print(s.view(3, 3))
                print(f"{winner} won!")
                return
                
        print("draw!")

if __name__ == '__main__':
    while True:
        print("Start train: 1\nStart Test: 2")
        x = input().strip()
        if x == "1":
            train(model)
        elif x == '2':
            TestTicTacToeModel(model)






