import  numpy as np
xall_new1 = [[1,2],[3,4],[5,6],[7,8],[9,10],[11,12]]
y_train2 = [0,1,1,0,1,0]
ordered_label_errors = [False,False,True,True,True,False]
y_train2 = np.array(y_train2)
xall_new1 = np.array(xall_new1)

# 创建一个条件，保留 y_train2 标签为0的数据
condition_0 = (y_train2 == 0)
print(y_train2 == 1)
print(ordered_label_errors)
# 创建一个条件，保留 y_train2 标签为1且在 x_mask 中保留的数据
condition_1 = (y_train2 == 1) & ordered_label_errors

# 将两个条件合并成一个总条件
total_condition = condition_0 | condition_1

# 应用总条件生成 x_mask
x_mask = total_condition

# 从 xall_new1 和 y_train2 中提取符合条件的数据
x_pruned = xall_new1[x_mask]
s_pruned = y_train2[x_mask]

print(x_pruned)