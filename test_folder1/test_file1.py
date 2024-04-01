def print_statement1():
    print("printing from test_file1")


from test_folder2 import test_file2 as tf2


def call_tf2_print():
    print("calling test_file2")
    tf2.print_statement2()



from test_folder3 import test_file3 as tf3

def call_tf3_print():
    tf3.print_statement3()
