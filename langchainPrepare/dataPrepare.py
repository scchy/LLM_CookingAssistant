# python3
# Create Date: 2024-01-10
# Author: Scc_hy
# Func: 数据预处理
# ==============================================================================

import json


def one_cooking_prepare(res_0):
    #  'title', 'intro', 'steps', 'ingredients', 'tags', 'notice', 'level', 'craft', 'duration', 'flavor'
    intr0_str = ',简介：' + res_0['intro'] if len(res_0['intro']) else ""
    f_str = res_0['title'] + intr0_str + "。需要准备材料：{}。总共需要{}个主要步骤，需要{}, 步骤如下：\n{}".format(
        ','.join([f'{k}:{v}' for k, v in res_0['ingredients'].items()]),
        len(res_0['steps']),
        res_0['duration'],
        ''.join([f'{stp["index"]}- {stp["content"].replace("疑惑", "诱惑")}\n'  for stp in res_0['steps']])
    ) + "{},{}\n难度：{},口味：{}，烹饪方式：{}，耗时：{}".format(
        res_0['tags'],
        res_0['notice'],
        res_0['level'],
        res_0['flavor'],
        res_0['craft'],
        res_0['duration']
    )
    return  f_str


def main():
    js_f = 'mstx-中文菜谱.json'
    with open(js_f, 'r') as f:
        res = json.load(f)
        
    final_js_list = [
        {'content': one_cooking_prepare(cook)} for cook in res
    ]
    print(len(final_js_list), len(res))
    out_f = 'cookingBook.json'
    with open(out_f, 'w') as f:
        json.dump(final_js_list, f)

    print('Finished prepare data')
    # check
    with open(out_f, 'r') as f:
        res = json.load(f)
    
    print('check load: {}\nres[0]={}'.format(len(res), str(res[0])))
    
    
if __name__ == '__main__':
    main()


