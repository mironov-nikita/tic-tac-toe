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

optimizer_fn = torch.optim.SGD(
    params=model.parameters(),
    lr=0.01) 


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


def TicTacToeGame(model):
    r = None
    s = torch.tensor([0,0,0,0,0,0,0,0,0], dtype=torch.float32)
    MovesHistory = []
    for i in range(9):
        logits = model(s.unsqueeze(0))
        logits = logits[0]
        mask = (s == 0)                      
        logits_masked = logits.clone()
        logits_masked[~mask] = -float('inf')  
        move_index = torch.argmax(logits_masked)
        move_value = logits_masked[move_index]
        MovesHistory.append((move_index, move_value))

        s_new = s.clone()
        if i & 1:
            s_new[move_index] = -1
        else:
            s_new[move_index] = 1
        s = s_new  

        Winner = WhoHasWon(s)
        if not Winner:
            continue
        if Winner == "Tic":
            r = 1
            break
        else:
            r = -1
            break
    return r, MovesHistory


def MyLoss(MovesHistoryTic, MovesHistoryToe, r, gamma=0.9):
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


def train(model, epochs=10):
    model.train()
    print("========== TRAINING STARTS ==========")
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(10):
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
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
    s = torch.tensor([0,0,0,0,0,0,0,0,0], dtype=torch.float32)  
    model.eval()
    with torch.inference_mode():
        n = 9
        FirstMove = random.randint(1, 2)
        if FirstMove == 1:
            logits = model(s.unsqueeze(0))  
            logits = logits[0]  
            mask = (s == 0)
            logits[~mask] = -float('inf')
            move_index = torch.argmax(logits).item()
            
            s[move_index] = 1
            j = move_index // 3
            i = move_index % 3
            board[j][i] = 1
            n -= 1
        PlayerFlag = 0
        if n == 8:
            print("You play with minus ones")
            PlayerFlag = -1
        else:
            print("You play with ones")
            PlayerFlag = 1
        for _ in range(n//2):
            for row in board:
                print(row)
            print("Your turn. To make a move, write a number from 1 to 9.")
            PlayerMove = input("").strip()
            if PlayerMove not in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                print('Enter correctly, otherwise the game will end.')
                PlayerMove = input("").strip()
                if PlayerMove not in ('1', '2', '3', '4', '5', '6', '7', '8', '9'):
                    print("You entered the coordinate incorrectly the second time.")
                    return 
                
            PlayerMove = int(PlayerMove) - 1  
            j = PlayerMove // 3
            i = PlayerMove % 3
            board[j][i] = PlayerFlag
            s[PlayerMove] = PlayerFlag
            Winner = WhoHasWon(s)
            if Winner == "Tic":
                for row in board:
                    print(row)
                print("The Tic won!")
                return
            elif Winner == "Toe":
                for row in board:
                    print(row)
                print("The Toe won!")
                return

            logits = model(s.unsqueeze(0))[0]
            mask = (s == 0)
            logits_masked = logits.clone()
            logits_masked[~mask] = -float('inf')
            move_index = torch.argmax(logits_masked).item()
            
            j = move_index // 3
            i = move_index % 3
            board[j][i] = PlayerFlag * -1
            s[move_index] = PlayerFlag * -1
            
            Winner = WhoHasWon(s)
            if Winner == "Tic":
                for row in board:
                    print(row)
                print("The Tic won!")
                return
            elif Winner == "Toe":
                for row in board:
                    print(row)
                print("The Toe won!")
                return
            
        for row in board:
            print(row)
        print("1/2 | 1/2")
        print("= Draw! =")

if __name__ == '__main__':
    x = input("Start train: 1\nStart Test: 2\n").strip()
    if x == "1":
        train(model)
    elif x == '2':
        TestTicTacToeModel(model)




