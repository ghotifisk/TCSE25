# Page.19 Rice and Chessboard Problem

s = 0; n = 1; fv = 1
while n < 65:
    s = s + fv
    n = n +1
    fv = 2 * fv
print(s)