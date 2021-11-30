# 基尼系数
def gini(groups, classes):
    global_gini = 0.0
    total_instances = sum([len(instance) for instance in groups])
    for group in groups:
        local_gini = 0.0
        group_size = len(group)
        if group_size == 0:
            continue
        for class_ in classes:
            prob = [row[-1] for row in group].count(class_) / group_size
            local_gini += prob * (1 - prob)
        global_gini += local_gini * (group_size / total_instances)
    return global_gini


# 为了计算基尼指数，我们要把当前的数据集进行划分（CART算法生成的是二叉树，只需把数据集划分为两个子集即可）
def split(dataset, index, value):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right


# 遍历所有的特征值，根据数据集的划分来计算其基尼指数
def get_min_gini(dataset):
    classes = list(set([instance[-1] for instance in dataset]))
    features = len(dataset[0]) - 1
    index, value, score, group = 999, 999, 999, None
    for feature in range(features):
        feature_values = set([instance[feature] for instance in dataset])
        for feature_value in feature_values:
            group = split(dataset, feature, feature_value)
            gini_score = gini(group, classes)
            if gini_score < score:
                index, value, score, group = feature, feature_value, gini_score, group
    return {'index': index, 'value': value, 'group': group}


# 利用叶结点中出现次数最多的类来代表叶结点
def leaf_node(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)


def Start(dataset, limit_depth, limit_group):
    node = get_min_gini(dataset)
    depth = 1
    Build_Tree(node, depth, limit_depth, limit_group)
    return node


# 递归：生成树
def Build_Tree(node, depth, limit_depth, limit_group):
    left, right = node['group']
    del (node['group'])
    print(node)
    if not left or not right:
        node['left'] = node['right'] = leaf_node(left + right)
        return
    if depth >= limit_depth:
        node['left'] = leaf_node(left)
        node['right'] = leaf_node(right)
        return
    if len(left) <= limit_group:
        node['left'] = leaf_node(left)
    else:
        node['left'] = get_min_gini(left)
        Build_Tree(node['left'], depth + 1, limit_depth, limit_group)
    if len(right) <= limit_group:
        node['right'] = leaf_node(right)
    else:
        node['right'] = get_min_gini(right)
        Build_Tree(node['right'], depth + 1, limit_depth, limit_group)


def print_tree(node, depth=0):
    if isinstance(node, dict):
        print('%s[X%d < %.3f]' % ((depth * ' ', (node['index'] + 1), node['value'])))
        print_tree(node['left'], depth + 1)
        print_tree(node['right'], depth + 1)
    else:
        print('%s[%s]' % ((depth * ' ', node)))


dataset = [[2.771244718, 1.784783929, 0],
           [1.728571309, 1.169761413, 0],
           [3.678319846, 2.81281357, 0],
           [3.961043357, 2.61995032, 0],
           [2.999208922, 2.209014212, 0],
           [7.497545867, 3.162953546, 1],
           [9.00220326, 3.339047188, 1],
           [7.444542326, 0.476683375, 1],
           [10.12493903, 3.234550982, 1],
           [6.642287351, 3.319983761, 1]]
tree = Start(dataset, 2, 1)
print_tree(tree)
