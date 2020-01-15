from django.shortcuts import render
from django.contrib import messages
from django.conf import settings
import io
import csv
import pandas
from app.nb_model import model

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
                print(coulmn)
                data.append({"label":coulmn[1],"text":coulmn[0]})
                data_df.append([coulmn[1],coulmn[0]])
            
            # print(data_df)
            df = pandas.DataFrame(data_df,columns=["label","text"])
            df = df.dropna()
            df.reset_index()
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

    # try:
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
    
        # try:
    data = []
    data_df = []
    i = 1

    model.training()

    for column in csv.reader(io_string,delimiter=',', quotechar="|"):
        _ , y_pred = model.testing(column[0],column[1])
        data.append({"label":column[1],"text":column[0],"is_valid":column[2],
        "text_token": column[0].split(" "),
        "text_vector":model.text_to_vector(column[0]),
        "y_pred" : y_pred[0],
        "index": i
        
        })
        data_df.append([column[1],column[0],column[2]])
        i += 1

    df = pandas.DataFrame(data_df,columns=["label","text","is_valid"])
    df = df.dropna()
    df.reset_index()
    df.to_csv("./static/datatest.csv", sep=',', encoding='utf-8',index=None)

    
    acc , y_pred = model.testing()
    context = {"data": data,
                "accuracy_val": acc[0]*100,
                "accuracy_test": acc[1]*100 }
    return render(request, template, context)
    #     except:
    #         propt = {
    #             'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
    #             'err_info' : "error : the seperator file is not consistent"
    #         }
            
    #         return render(request, template, propt)
    # except:
    #     propt = {
    #             'info' : '* only allowed file with csv extention, with coma seperator, header true, 3 column [label,text,is_valid]',
    #             'err_info' : "error : please choose file before submit"
    #         }
            
    #     return render(request, template, propt)