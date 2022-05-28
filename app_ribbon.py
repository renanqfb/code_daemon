

#%%

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



close = list(range(100,139,1))
close.extend(list(range(139,84,-1)))

close = pd.DataFrame({'PX LAST':close})
close['HIGH'] = close['PX LAST'] + 1
close['LOW'] = close['PX LAST'] - 1
close['ASSET'] = 'Test Index'
close['DATES'] = pd.Timestamp(1900,1,1)
date_inicial = pd.Timestamp(1989,1,1)
for i in range(len(close)):
    close['DATES'].iloc[i] = date_inicial
    date_inicial = date_inicial + pd.DateOffset(days=1)


# Criando funções para analisar o MA Ribbon
def move_averages(data_frame, type_ma,ma,name,sinal,wg,inc):

    import statistics as st
    mv=[]
    for i in range(0,10):
        mv.append(ma+i*inc)
    if type_ma=='Simple':
        for j in range(len(mv)):
            data_frame['ma'+str(mv[j])]=0
            for i in range(len(data_frame)):
                if i<mv[j]:
                    data_frame['ma'+str(mv[j])].iloc[i]=data_frame[name].iloc[0:mv[j]].mean()
                else:
                    n=i-mv[j]+1
                    mov=i+1
                    data_frame['ma'+str(mv[j])].iloc[i]=data_frame[name].iloc[n:mov].mean()
    elif type_ma=='Exponential':
        for j in range(len(mv)):
            data_frame['ma'+str(mv[j])]=0
            for i in range(len(data_frame)):
                if i<=mv[j]:
                    data_frame['ma'+str(mv[j])].iloc[i]=st.mean(data_frame[name].iloc[0:mv[j]])
                else:
                    data_frame['ma'+str(mv[j])].iloc[i]=data_frame[name].iloc[i]*(2/(mv[j]+1))+data_frame['ma'+str(mv[j])].iloc[i-1]*(1-(2/(mv[j]+1)))


    elif type_ma=='Smoothed':
        for j in range(len(mv)):
            data_frame['ma'+str(mv[j])]=0
            for i in range(len(data_frame)):
                if i<mv[j]:
                    data_frame['ma'+str(mv[j])].iloc[i]=data_frame[name].iloc[0:mv[j]].mean()
                else:
                    data_frame['ma'+str(mv[j])].iloc[i]=(data_frame['ma'+str(mv[j])].iloc[i-1]*(mv[j]-1)+data_frame[name].iloc[i])/mv[j]



    data_frame['min']=0
    data_frame['max']=0
    data_frame['signal']=0

    if sinal=='Simple':
        for i in range(len(data_frame)):
            data_frame['min'].iloc[i]=min(data_frame.iloc[i,3:13])
            data_frame['max'].iloc[i]=max(data_frame.iloc[i,3:13])
            if data_frame[name].iloc[i]>data_frame['max'].iloc[i]:
                data_frame['signal'].iloc[i]=1
            elif data_frame[name].iloc[i]<data_frame['min'].iloc[i]:
                data_frame['signal'].iloc[i]=-1
            else:
                data_frame['signal'].iloc[i]=0
    elif sinal=='Simple_continue':

        data_frame['inclination']=0
        for i in range(20,len(data_frame)):

            incli10=(data_frame.iloc[i,12]-data_frame.iloc[i-20,12])/data_frame['tr'].iloc[i-20:i].mean()
            if incli10>0:
                data_frame['inclination'].iloc[i]=1
            elif incli10<0:
                data_frame['inclination'].iloc[i]=-1

        for i in range(len(data_frame)):
            data_frame['min'].iloc[i]=min(data_frame.iloc[i,3:13])
            data_frame['max'].iloc[i]=max(data_frame.iloc[i,3:13])
            if i<20:
                if data_frame[name].iloc[i]>data_frame['max'].iloc[i]:
                    data_frame['signal'].iloc[i]=1
                elif data_frame[name].iloc[i]<data_frame['min'].iloc[i]:
                    data_frame['signal'].iloc[i]=-1
                else:
                    data_frame['signal'].iloc[i]=0
            else:
                if data_frame[name].iloc[i]>data_frame['max'].iloc[i]:
                    data_frame['signal'].iloc[i]=1
                elif data_frame[name].iloc[i]<data_frame['min'].iloc[i]:
                    data_frame['signal'].iloc[i]=-1
                elif data_frame[name].iloc[i]<data_frame['max'].iloc[i] and data_frame[name].iloc[i]>data_frame['min'].iloc[i] and data_frame['signal'].iloc[i-1]==1 and data_frame['inclination'].iloc[i]==1:
                    data_frame['signal'].iloc[i]=1
                elif data_frame[name].iloc[i]<data_frame['max'].iloc[i] and data_frame[name].iloc[i]>data_frame['min'].iloc[i]  and data_frame['signal'].iloc[i-1]==-1 and data_frame['inclination'].iloc[i]==-1:
                    data_frame['signal'].iloc[i]=-1
                else:
                    data_frame['signal'].iloc[i]=0
    elif sinal=='Score_old':
        data_frame['inclination']=0
        data_frame['score']=0
        for i in range(20,len(data_frame)):
            incli10=(data_frame.iloc[i,7]-data_frame.iloc[i-1,7])/data_frame['tr'].iloc[i-1:i].mean()
            if incli10>0:
                data_frame['inclination'].iloc[i]=1
            elif incli10<0:
                data_frame['inclination'].iloc[i]=-1

        for i in range(0,len(data_frame)):
            score=0
            for j in range(3,13):
                if j==3:
                    point=0.25
                elif j==4:
                    point=0.15
                else:
                    point=0
                for g in range(j+1,13):
                    if data_frame.iloc[i,j]>data_frame.iloc[i,g]:
                        score=score+point
                    else:
                        score=score-point
            if data_frame['inclination'].iloc[i]==1:
                score=score+1
            elif data_frame['inclination'].iloc[i]==-1:
                score=score-1
            if data_frame[name].iloc[i]>data_frame.iloc[i,3]:
                score=score+1
            else:
                score=score-1
            if score<0:
                data_frame['signal'].iloc[i]=-1
            elif score>0:
                data_frame['signal'].iloc[i]=1
            else:
                data_frame['signal'].iloc[i]=0
            data_frame['score'].iloc[i]=score

    elif sinal=='Score_new':
        data_frame['inclination']=0
        data_frame['score']=0
        point=1
        incli_0=0
        incli_3=0
        incli_6=0
        incli_9=0



        for i in range(0,len(data_frame)):
            score_tot=0
            cross_score=0
            for j in range(3,7):
                score=0
                for g in range(j+1,j+1+4):
                    if data_frame.iloc[i,j]>data_frame.iloc[i,g]:
                        score=score+point
                    else:
                        score=score-point
                cross_score=cross_score+(score/4)

            data_frame['min'].iloc[i]=min(data_frame.iloc[i,3:13])
            data_frame['max'].iloc[i]=max(data_frame.iloc[i,3:13])
            if data_frame[name].iloc[i]>data_frame['max'].iloc[i]:
                envelope_score=-2
            elif data_frame[name].iloc[i]<data_frame['min'].iloc[i]:
                envelope_score=2
            else:
                envelope_score=0
            if i>0:
                incli_0=data_frame.iloc[i,3]-data_frame.iloc[i-1,3]
                incli_3=data_frame.iloc[i,6]-data_frame.iloc[i-1,6]
                incli_6=data_frame.iloc[i,9]-data_frame.iloc[i-1,9]
                incli_9=data_frame.iloc[i,12]-data_frame.iloc[i-1,12]
            if incli_0>0:
                incli_0=1
            elif incli_0<0:
                incli_0=-1
            else:
                incli_0=0

            if incli_3>0:
                incli_3=1
            elif incli_3<0:
                incli_3=-1
            else:
                incli_3=0

            if incli_6>0:
                incli_6=1
            elif incli_6<0:
                incli_6=-1
            else:
                incli_6=0

            if incli_9>0:
                incli_9=1
            elif incli_9<0:
                incli_9=-1
            else:
                incli_9=0

            score_tot=envelope_score+cross_score+incli_0+incli_3+incli_6+incli_9

            if score_tot>0:
                data_frame['signal'].iloc[i]=1
                data_frame['score'].iloc[i]=score_tot
            elif score_tot<0:
                data_frame['signal'].iloc[i]=-1
                data_frame['score'].iloc[i]=score_tot
            else:
                data_frame['signal'].iloc[i]=0
                data_frame['score'].iloc[i]=score_tot



    return data_frame




def plot_ma_ribbon(data_frame,ma,name,type_ma,inc):
    import matplotlib.pyplot as plt
    mv=[]
    for i in range(0,10):
        mv.append('ma'+str(ma+i*inc))
    data_buy=data_frame[data_frame['signal']==1]
    data_neutral=data_frame[data_frame['signal']==0]
    data_sell=data_frame[data_frame['signal']==-1]
    plt.figure(figsize=(15,6))
    plt.title('Moving Average '+'- '+type_ma)
    plt.plot(data_frame[mv[0]],label=mv[0])
    plt.plot(data_frame[mv[1]],label=mv[1])
    plt.plot(data_frame[mv[2]],label=mv[2])
    plt.plot(data_frame[mv[3]],label=mv[3])
    plt.plot(data_frame[mv[4]],label=mv[4])
    plt.plot(data_frame[mv[5]],label=mv[5])
    plt.plot(data_frame[mv[6]],label=mv[6])
    plt.plot(data_frame[mv[7]],label=mv[7])
    plt.plot(data_frame[mv[8]],label=mv[8])
    plt.plot(data_frame[mv[9]],label=mv[9])
    plt.scatter(data_buy.index,data_buy[name],label='buy',c='green')
    plt.scatter(data_neutral.index,data_neutral[name],label='neutral',c='black')
    plt.scatter(data_sell.index,data_sell[name],label='sell',c='red')
    plt.legend(loc='upper left')
    plt.show()


def tr_and_n(df_c_h_l,data_frame,name):
    import statistics as st
    data_frame['tr']=0
    data_frame['n']=0
    df_c_h_l=df_c_h_l[df_c_h_l['ASSET']==name]

    for i in range(len(data_frame)):

        if i<20:
           data_frame['tr'].iloc[i]= df_c_h_l['HIGH'].iloc[i]-df_c_h_l['LOW'].iloc[i]
           if i ==0:
              data_frame['n'].iloc[i]=data_frame['tr'].iloc[i]
           else:
              data_frame['n'].iloc[i]=st.mean(data_frame['tr'].iloc[:i+1])
        else:
            data_frame['tr'].iloc[i]= max(abs(df_c_h_l['HIGH'].iloc[i]-df_c_h_l['LOW'].iloc[i]),abs(df_c_h_l['HIGH'].iloc[i]-df_c_h_l['PX LAST'].iloc[i-1]),abs(df_c_h_l['PX LAST'].iloc[i-1]-df_c_h_l['LOW'].iloc[i]))
            data_frame['n'].iloc[i]=((20-1)*data_frame['n'].iloc[i-1]+data_frame['tr'].iloc[i])/20
    return data_frame


def models_ma_ribbon(data_frame,type_model,name,final_ma):
    if type_model=='Only_buy':
        data_frame['pct_change']=data_frame[name].pct_change()
        data_frame['weight']=0
        for i in range(len(data_frame)):
            if data_frame['signal'].iloc[i-1]==1:
                data_frame['weight'].iloc[i]=1
    if type_model=='Only_sell':
        data_frame['pct_change']=data_frame[name].pct_change()
        data_frame['weight']=0
        for i in range(len(data_frame)):
            if data_frame['signal'].iloc[i-1]==-1:
                data_frame['weight'].iloc[i]=-1
    elif type_model=='Ribbon':
        data_frame['pct_change']=data_frame[name].pct_change()
        data_frame['weight']=0
        for i in range(1,len(data_frame)):
            if data_frame['signal'].iloc[i]==1:
                data_frame['weight'].iloc[i]=1
            elif data_frame['signal'].iloc[i]==-1:
                data_frame['weight'].iloc[i]=-1
    elif type_model=='Buy_sell':
        data_frame['pct_change']=data_frame[name].pct_change()
        data_frame['weight']=0
        for i in range(1,len(data_frame)):
            if data_frame['signal'].iloc[i-1]==1:
                data_frame['weight'].iloc[i]=1
            elif data_frame['signal'].iloc[i-1]==-1:
                data_frame['weight'].iloc[i]=-1
            else:
                data_frame['weight'].iloc[i]=data_frame['weight'].iloc[i-1]
    elif type_model == 'Buy_sell_with_stop':
        data_frame['weight']=0
        data_frame['pct_change']=data_frame[name].pct_change()
        data_frame['stop']=0
        test=0
        for i in range(1,len(data_frame)):
            if test==0:
                if data_frame['signal'].iloc[i-1]==-1:
                   data_frame['weight'].iloc[i]=-1
                   data_frame['stop'].iloc[i]=data_frame[name].iloc[i-1]+2*data_frame['n'].iloc[i-1]
                   test=1
                elif data_frame['signal'].iloc[i-1]==1:
                    data_frame['weight'].iloc[i]=1
                    data_frame['stop'].iloc[i]=data_frame[name].iloc[i-1]-2*data_frame['n'].iloc[i-1]
                    test=1
                else:
                    data_frame['weight'].iloc[i]=0
                    test=0
            else:
                if data_frame['weight'].iloc[i-1]==-1:
                    data_frame['weight'].iloc[i]=-1
                    if data_frame[name].iloc[i]>data_frame['stop'].iloc[i-1]:
                        test=0
                    else:
                        if data_frame[name].iloc[i]<data_frame[name].iloc[i-1]-(1/2)*data_frame['n'].iloc[i-1]:
                            data_frame['stop'].iloc[i]=data_frame['stop'].iloc[i-1]-(1/2)*data_frame['n'].iloc[i-1]
                        else:
                            data_frame['stop'].iloc[i]=data_frame['stop'].iloc[i-1]
                else:
                    data_frame['weight'].iloc[i]=1
                    if data_frame[name].iloc[i]<data_frame['stop'].iloc[i-1]:
                        test=0
                    else:
                        if data_frame[name].iloc[i]>data_frame[name].iloc[i-1]+(1/2)*data_frame['n'].iloc[i-1]:
                            data_frame['stop'].iloc[i]=data_frame['stop'].iloc[i-1]+(1/2)*data_frame['n'].iloc[i-1]
                        else:
                            data_frame['stop'].iloc[i]=data_frame['stop'].iloc[i-1]


    data_frame=data_frame.iloc[final_ma:]
    data_frame['returns']=data_frame['weight']*data_frame['pct_change']

    return data_frame

tipo_ma='Smoothed'
sinal_type='Score_new'
tipo_model='Ribbon'
name='Test Index'
wg=0
ma=10
inc=5

final_ma=ma+9*inc

data=close[close['ASSET']==name]
data=data.pivot_table(index='DATES',columns='ASSET',values='PX LAST')

close_high_low=close[close['ASSET']==name]
ma_data=tr_and_n(close_high_low,data,name)
ma_data=move_averages(ma_data,tipo_ma,ma,name,sinal_type,wg,inc)
img = plot_ma_ribbon(ma_data,ma,name,tipo_ma,inc)

selection=pd.DataFrame({'weight':ma_data['signal'],
                        'price':ma_data[name],
                        'n':ma_data['n']})

j=0
selection['P&L']=0
selection['P&L Cum']=0
selection['Position']=0
wgt=0
for i in range(len(selection)):
   if j==0:
       wgt=selection['weight'].iloc[i]
       selection['Position'].iloc[i]=(100000/selection['n'].iloc[i])*selection['weight'].iloc[i]
       j=1
   else:
       if wgt != selection['weight'].iloc[i]:
           wgt=selection['weight'].iloc[i]
           selection['Position'].iloc[i]=(100000/selection['n'].iloc[i])*wgt
           selection['P&L'].iloc[i]=selection['Position'].iloc[i-1]*(selection['price'].iloc[i]-selection['price'].iloc[i-1])
           selection['P&L Cum'].iloc[i]=selection['P&L Cum'].iloc[i-1]+selection['P&L'].iloc[i]
       else:
           selection['Position'].iloc[i]=selection['Position'].iloc[i-1]
           selection['P&L'].iloc[i]=selection['Position'].iloc[i-1]*(selection['price'].iloc[i]-selection['price'].iloc[i-1])
           selection['P&L Cum'].iloc[i]=selection['P&L Cum'].iloc[i-1]+selection['P&L'].iloc[i]

print(selection)

st.header('MA - Ribbon - Upon')

st.write('MA Ribbon signals:',img)
