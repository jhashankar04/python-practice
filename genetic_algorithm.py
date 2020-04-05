from random import randint, sample

n = int(input())
#n = 4

def board_generator(n):
    board = []
    for i in range(n):
        line = []
        for j in range(n):
            line.append(randint(1,4))
        board.append(line)
    return board

def evaluate(board, size=4): 
    invalid_instance = 0
    for i in range(n):
        for j in range(n):
            top = None
            bottom = None
            left = None
            right = None
            if i == 0:
                bottom = board[i+1][j]
                if j == 0:
                    right  = board[i][j+1]
                elif j == size - 1:
                    left = board[i][j-1]
                else: 
                    right  = board[i][j+1]
                    left = board[i][j-1]

            elif i == size - 1:
                top = board[i-1][j]
                if j == 0:
                    right  = board[i][j+1]
                elif j == size - 1:
                    left = board[i][j-1]
                else: 
                    right  = board[i][j+1]
                    left = board[i][j-1]
            else:
                top = board[i-1][j]
                bottom = board[i+1][j]
                
                if j == 0:
                    right  = board[i][j+1]  
                elif j == size - 1:
                    left = board[i][j-1]
                else:
                    right  = board[i][j+1]
                    left = board[i][j-1]

            in_correct_flag = int(board[i][j] in (top, bottom, left, right))
            invalid_instance = invalid_instance + in_correct_flag
    return(invalid_instance)
    
def select_best_k_boards(boards, size=4, k = 10): 
    board_and_scores = []
    for board in boards:
        board_and_scores.append((board,evaluate(board, size)))

    board_and_scores.sort(key=lambda x:x[1])
    selected_boards_and_scores = board_and_scores[:k].copy()
        
    return selected_boards_and_scores

def reproduce(selected_best_boards, number_of_boards = 90, size=4):
    new_boards = []
    for i in range(number_of_boards):
        row_index = sample(range(size), k =size)
        boards_index  = sample(range(10), size)
        baby_board = []
        for j in range(0,len(row_index)):
            baby_board.append(selected_best_boards[boards_index[j]][row_index[j]])
        new_boards.append(baby_board)
    new_boards.extend(selected_best_boards)
    return(new_boards)


boards = []
# initialize
for i in range(100):
    boards.append(board_generator(n))

selected_boards_and_scores = select_best_k_boards(boards, size=n)
selected_boards = list(map(lambda x:x[0],selected_boards_and_scores))
selected_scores = list(map(lambda x:x[1],selected_boards_and_scores))
best_score = selected_scores[0]
best_board = selected_boards[0]

#print(best_score)
#best_score = 100000
while best_score != 0:
    new_boards = reproduce(selected_boards, size = n)
    selected_boards_and_scores = select_best_k_boards(new_boards, size=n)
    selected_boards = list(map(lambda x:x[0],selected_boards_and_scores))
    selected_scores = list(map(lambda x:x[1],selected_boards_and_scores))
    best_score = selected_scores[0]
    best_board = selected_boards[0]
    #print(best_score)

for row in best_board:
    print(" ".join(map(str,row)))

