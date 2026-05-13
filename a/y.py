from graphviz import Digraph

# 设置更小的间距参数
dot = Digraph(comment='Compact Irrigation', format='png')
dot.attr(rankdir='TB', size='6,8') # 改为纵向布局
dot.attr(nodesep='0.2', ranksep='0.3') # 极度压缩节点和层级间的距离
dot.attr('node', fontname='Microsoft YaHei', fontsize='10', 
         shape='rect', style='filled, rounded', 
         margin='0.1,0.05', # 减小文字与边框的距离
         height='0.3', width='1.0') # 统一并缩小节点尺寸

# 定义节点（颜色更高级一点）
dot.node('S', '开始', fillcolor='#D5D8DC', color='#566573')
dot.node('A', '采集数据', fillcolor='#AED6F1', color='#2E86C1')
dot.node('B', '加权融合', fillcolor='#AED6F1', color='#2E86C1')
dot.node('C', '达标？', shape='diamond', fillcolor='#F9E79F', color='#D4AC0D')
dot.node('D', '启动灌溉', fillcolor='#ABEBC6', color='#28B463')
dot.node('E', '监测指数', fillcolor='#ABEBC6', color='#28B463')
dot.node('F', '停止？', shape='diamond', fillcolor='#F9E79F', color='#D4AC0D')
dot.node('G', '停止灌溉', fillcolor='#F2D7D5', color='#CB4335')

# 紧凑连线
dot.edge('S', 'A')
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D', label='是')
dot.edge('C', 'A', label='否')
dot.edge('D', 'E')
dot.edge('E', 'F')
dot.edge('F', 'G', label='是')
dot.edge('F', 'E', label='否')
dot.edge('G', 'A', label='循环')

dot.render('compact_flow', view=True)