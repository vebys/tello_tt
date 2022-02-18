import os
wait_path = '../tt_data/train/labels/wait_change/'
finish_path = '../tt_data/train/labels/finish_change/'


wait_file = os.listdir(wait_path)
j=1
for w_f in wait_file:
    content = ''
    with open(f"{wait_path}{w_f}") as f:
       for i in  f.readlines():
            # print(type(i[:1]))
            code = '1' if i[:1] =='0' else '0'
            content= f"{content}{code}{i[1:]}"
    with open(f'{finish_path}{w_f}','w') as g:
        g.write(content)
    # if j>4:
    #     break
    # j+=1

