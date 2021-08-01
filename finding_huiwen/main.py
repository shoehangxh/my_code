import sys
def Load(file):
    try:
        # 处理文件打开异常
        with open(file) as in_file:
            # 保证文件会自动关闭
            load_txt = in_file.read().strip().split('\n')
            # 阅读、去除首尾空格、以回车为分割保存词至列表变量中
            load_txt = [x.lower() for x in load_txt]
            # 返回小写字母
            return load_txt
    except IOError as e:
        # 若没有这个类型的错误，程序就只执行try语句，如果有其他类型的错误，会抛出try结构体
        print('{}\nError opening {}. Terminating program.'.format(e, file), file=sys.stderr)
        sys.exit(1)
        # 程序终止，1代表异常退出