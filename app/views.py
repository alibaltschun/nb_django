from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
import io
import csv
import pandas
from app import model

def train(request):
    template = 'train_page.html'
    
    propt = {
        'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
        'err_info' : ''
    }

    if request.method == "GET":
        return render(request, template, propt)

    try:
        csv_file = request.FILES['file']

        if not csv_file.name.endswith('.csv'):
            propt = {
                'info' : '* only allowed file with csv extention, with coma seperator, header true, 2 column [label,text]',
                'err_info' : "error : the extention file that you submit is not a csv"
            }
            
            return render(request, template, propt)

        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string)
        try:
            data = []
            data_df = []
            for coulmn in csv.reader(io_string,delimiter=',', quotechar="|"):
                data.append({"label":coulmn[0],"text":coulmn[1]})
                data_df.append([coulmn[0],coulmn[1]])
            
            # print(data_df)
            df = pandas.DataFrame(data_df,columns=["label","text"])
            # print(df)
            df.to_csv("./static/datatrain.csv", sep=',', encoding='utf-8',index=None)
            context = {"data": data}
            return render(request, template, context)
        except:
            propt = {
                'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
                'err_info' : "error : the seperator file is not consistent"
            }
            
            return render(request, template, propt)
    except:
        propt = {
                'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
                'err_info' : "error : please choose file before submit"
            }
            
        return render(request, template, propt)


def test(request):
    template = 'test_page.html'
    
    propt = {
        'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
        'err_info' : ''
    }

    if request.method == "GET":
        return render(request, template, propt)

    try:
        csv_file = request.FILES['file']

        if not csv_file.name.endswith('.csv'):
            propt = {
                'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
                'err_info' : "error : the extention file that you submit is not a csv"
            }
            
            return render(request, template, propt)

        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        next(io_string)
        try:
            data = []
            data_df = []
            i = 1
            for coulmn in csv.reader(io_string,delimiter=',', quotechar="|"):
                _ , y_pred = model.testing(coulmn[1],coulmn[0])
                data.append({"label":coulmn[0],"text":coulmn[1],"is_valid":coulmn[2],
                "text_token":model.preprocessingTokenization(coulmn[1]),
                "text_vector":model.preprocessingVectorizer([coulmn[1]]),
                "y_pred" : y_pred[0],
                "index": i
                
                })
                data_df.append([coulmn[0],coulmn[1],coulmn[2]])
                i += 1
            print(1)
            df = pandas.DataFrame(data_df,columns=["label","text","is_valid"])
            df.to_csv("./static/datatest.csv", sep=',', encoding='utf-8',index=None)

            model.training()
            acc , y_pred = model.testing()
            print(2)
            context = {"data": data,
                       "accuracy_val": acc[0]*100,
                       "accuracy_test": acc[1]*100 }
            return render(request, template, context)
        except:
            propt = {
                'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
                'err_info' : "error : the seperator file is not consistent"
            }
            
            return render(request, template, propt)
    except:
        propt = {
                'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
                'err_info' : "error : please choose file before submit"
            }
            
        return render(request, template, propt)