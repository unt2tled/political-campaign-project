input = open("input_texts_170722.txt","r")
output = open("output_texts_170722.txt","w")
line_write = None
line_read = input.readline()
counter = 0
while (line_read):
    line_read = line_read.strip()
    if (line_read[-8:]=='text.txt'):
        counter+=1
        if (counter % 100 == 0):
            print(counter)
            print(line_read)
        if (line_write != None):
            output.write(line_write+'\n')
        line_write = line_read[:-8] +'*'
    else:
        line_write += " " + line_read
    line_read = input.readline()
input.close()
if (line_write != None):
    output.write(line_write+'\n')
output.close()

    
