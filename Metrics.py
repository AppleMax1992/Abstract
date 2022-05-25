from rouge import Rouge
import pandas as pd
from tqdm.contrib import tzip
# 计算词粒度的rouge-1、rouge-2、rouge-L
rouge = Rouge()
data = pd.read_csv('result.csv')
metric = pd.DataFrame(columns=['content','title','predict_title','rouge-1','rouge-2','rouge-l'])
R1_Precision = 0
R1_Recall = 0
R1_F1_Score = 0
R2_Precision = 0
R2_Recall = 0
R2_F1_Score = 0
RL_Precision = 0
RL_Recall = 0
RL_F1_Score = 0
for title,content,pred_tilte in tzip(data['title'],data['content'],data['predict_title']):
    pred = ' '.join(list(title))
    golden = ' '.join(list(pred_tilte))
    rouge_scores = rouge.get_scores(hyps=pred, refs=golden)
    metric.loc[len(metric.index)] = [content,title,golden,'p:'+str(rouge_scores[0]['rouge-1']['p'])+'\n'+'r:'+ str(rouge_scores[0]['rouge-1']['r'])+'\n'+ 'f:'+ str(rouge_scores[0]['rouge-1']['f']),'p:'+str(rouge_scores[0]['rouge-2']['p'])+'\n'+'r:'+ str(rouge_scores[0]['rouge-2']['r'])+'\n'+ 'f:'+ str(rouge_scores[0]['rouge-2']['f']),'p:'+str(rouge_scores[0]['rouge-l']['p'])+'\n'+'r:'+ str(rouge_scores[0]['rouge-l']['r'])+'\n'+ 'f:'+ str(rouge_scores[0]['rouge-l']['f'])]
    R1_Precision = R1_Precision + rouge_scores[0]['rouge-1']['p']
    R1_Recall = R1_Recall + rouge_scores[0]['rouge-1']['r']
    R1_F1_Score = R1_F1_Score + rouge_scores[0]['rouge-1']['f']
    R2_Precision = R2_Precision + rouge_scores[0]['rouge-2']['p']
    R2_Recall = R2_Recall + rouge_scores[0]['rouge-2']['p']
    R2_F1_Score = R2_F1_Score + rouge_scores[0]['rouge-2']['p']
    RL_Precision = RL_Precision +rouge_scores[0]['rouge-l']['p']
    RL_Recall = RL_Recall +rouge_scores[0]['rouge-l']['r']
    RL_F1_Score = RL_F1_Score +rouge_scores[0]['rouge-l']['f']
    # print(rouge_scores)
# print(R1_Precision)
# print(len(data))
print('R1-Average-Precision:'+str(R1_Precision/len(data)*100)[:5]+'%')
print('R1-Average-Recall:'+str(R1_Recall/len(data)*100)[:5]+'%')
print('R1-Average-F1_Score:'+str(R1_F1_Score/len(data)*100)[:5]+'%')
print('R2-Average-Precision:'+str(R2_Precision/len(data)*100)[:5]+'%')
print('R2-Average-Recall:'+str(R2_Recall/len(data)*100)[:5]+'%')
print('R2-Average-F1_Score:'+str(R2_F1_Score/len(data)*100)[:5]+'%')
print('RL-Average-Precision:'+str(RL_Precision/len(data)*100)[:5]+'%')
print('RL-Average-Recall:'+str(RL_Recall/len(data)*100)[:5]+'%')
print('RL-Average-F1_Score:'+str(RL_F1_Score/len(data)*100)[:5]+'%')
metric.to_csv('res_with_metric.csv',encoding='utf_8_sig')