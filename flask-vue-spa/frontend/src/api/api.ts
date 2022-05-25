import { message } from "ant-design-vue";
import axios from "axios";
import localforage from 'localforage';
import { getBrowserKey } from './index';


const whiteList: Array<string> = [];
const commonReqInterceptors = async (config: any) => {
    const username = await window.sessionStorage.getItem('username');
    if (whiteList.indexOf(config.url) > -1) { // 有几个接口必须登录，否则不让访问
        if (username) {
            window.sessionStorage.clear();
            message.error('登录失效');
            window.location.href = '/';
        }
    } else {
        if (username) {
            config.headers['username'] = username;
        }

        const token = await window.sessionStorage.getItem('token');
        if (token) {
            config.headers['token'] = token;
        } else {
            const browserKey = await getBrowserKey();
            config.headers['token'] = browserKey;
        }
    }
    return config;
}

const commonRepInterceptors = async (res: any) => {
    if (res.data.code !== 200) {
        // message.info(res.data.msg);
        if (res.data.code === 401) {
            window.sessionStorage.clear();
            window.location.href = '/';
        }
    }

}






// ---账号相关---
const hwyIns = axios.create();
hwyIns.defaults.baseURL = '/hwy';
hwyIns.interceptors.request.use(
    async (config: any) => {
        const myConfig = await commonReqInterceptors(config);
        return myConfig;
    },
    (error) => {
        return Promise.reject(error)
    }
)
hwyIns.interceptors.response.use((res) => {
    commonRepInterceptors(res);
    return Promise.resolve(res)
}, (error) => {
    return Promise.reject(error);
});

// 注册
export const register = (params: any) => {
    return hwyIns.post('/register', params)
}
//获取角色分类
export const getRole = () => {
    return hwyIns.get('/rbac/role')
}
// 登录
export const login = (params: any) => {
    return hwyIns.post('/login', params)
}
// 短信登录
export const phone_login = (params: any) => {
    return hwyIns.post('/phone_login', params)
}
// 获取检索页数据
export const getSearchHistory = (params: any) => {
    return hwyIns.get('/historylist', params)
}
// 添加检索历史记录
export const addSearchHistory = (params: any) => {
    return hwyIns.post('/addhistory', params)
}
// 存取检索历史记录
export const deleteSearchHistory = (params: any) => {
    return hwyIns.delete('/delhistory', { params })
}


// 获取我的收藏列表
export const getColletlist = (params: any) => {
    return hwyIns.get('/colletlist', params)
}
// 删除我的收藏
export const deleteCollet = (params: any) => {
    return hwyIns.delete('/delcollet', { params })
}

// 重置密码
// 删除我的收藏
export const resetPasswd = (params: any) => {
    return hwyIns.put('/change_password', params)
}


export const revise_user = (params: any) => {
    return hwyIns.put('/revise_user', params)
}

export const revise_phone = (params: any) => {
    return hwyIns.put('/revise_phone', params)
}

export const getPhoneCode = (params: any) => {
    return hwyIns.post("/send_msg", params);
}
export const beSurePhoneCode = (params: any) => {
    return hwyIns.put("/send_msg", params);
}
export const getAddrecord = (params: any) => {
    return hwyIns.post("/addrecord", params);
}

export const getTypeList = () => {
    return hwyIns.get("/typelist");
}
// export const getQuestionDetail = (params: any) => {
//     return hwyIns.get(`/zhsf/question?question_id=${params}`);
// }
// ---其他相关---
const abcIns = axios.create();
abcIns.defaults.baseURL = '/abc';
abcIns.interceptors.request.use(
    async (config: any) => {
        const myConfig = await commonReqInterceptors(config);
        return myConfig;
    },
    (error) => {
        return Promise.reject(error)
    }
)
abcIns.interceptors.response.use((res) => {
    commonRepInterceptors(res);
    return Promise.resolve(res)
}, (error) => {
    return Promise.reject(error);
});


// 获取搜索热词
export const getHotWd = () => {
    return abcIns.get('/zhsf/hotquestion')
}

// 获取主页数据
export const getSearchData = (search: string) => {
    return abcIns.get(`/zhsf/head/left?msg=${search}`)
}
// 获取法律法规数据
export const getLawSearchData = (search: string) => {
    return abcIns.get(`/zhsf/head/laws?msg=${search}`)
}

// 获取案例数据
export const getCaseData = (params: any) => {
    // return abcIns.get("/mer/query?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size + "&province=" + params.provinces + "&city=" + params.citys);
    return abcIns.get("/mer/query?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size)
}

// 获取案例数据( 精排后 )
// export const getCaseData = (params: any) => {
//     // return abcIns.get("/mer/query?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size + "&province=" + params.provinces + "&city=" + params.citys);
//     return abcIns.get("/reorder/precise_search?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size)
// }
// 获取问答数据
export const getAnswerData = (params: any) => {
    // return abcIns.get("/mer/query?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size + "&province=" + params.provinces + "&city=" + params.citys);
    return abcIns.get("/zhsf/faq?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size + '&query=' + params.query)
}

// 获取问题推荐下拉数据
export const getQuestionList = (search: string) => {
    return abcIns.get(`/zhsf/questions?msg=${search}`)
}
// 获取问答推荐下拉数据
export const getQuestionPullList = (search: string) => {
    return abcIns.get(`/zhsf/questionspull?msg=${search}`)
}

export const getKGGraphList = (search: string) => {
    return abcIns.get(`/kg/recognize?msg=${search}`);
};

export const getNerList = (search: string) => {
    return abcIns.get(`/kg/ner?msg=${search}`);
};

export const getDetail = (params: any) => {
    return abcIns.get("/mer/cases?case_id=" + params.docId + "&msg=" + params.msg);
};

export const getLawDetail = (params: any) => {
    return abcIns.get("/zhsf/lawdetail?law_id=" + params.law_id + "&msg=" + params.msg);
};
//获取案例左侧列表数据
export const getstatistics = (params: any) => {
    return abcIns.get("/zhsf/cases/statistics?type=" + params.type
        + "&msg=" + params.msg
        // + params.startyear?"&startyear=" + params.startyear:''
        // + params.endyear ?"&endyear=" + params.endyear:''
        // + params.province?"&province=" + params.province:''
        // + params.city?"&city=" + params.city:''
        // + params.court?"&court=" + params.court:''
        // + params.docproperty?"&docproperty=" + params.docproperty:''
        // + params.casetype?"&casetype=" + params.casetype:'');
    )
};
// abcIns.interceptors.request.use(
//     async (config: any) => {
//         const res = await localforage.getItem('token');
//         if (res) {
//             config.headers['token'] = res;
//         } else {
//             localforage.clear();
//             message.error('登录失效');
//             window.location.href = '/login';
//         }
//         return config
//     },
//     (error) => {
//         return Promise.reject(error)
//     }
// )
// 获取检索报告数据
export const getReportData = (params: any) => {
    return abcIns.post('/zhsf/report', params)
}

// 获取检索页右边数据 /zhsf/head/right?msg
export const getSearchDataRight = (search: string) => {
    return abcIns.get(`/zhsf/head/right?msg=${search}`)
}

// 获取法规数据
export const getLawsData = (params: any) => {
    return abcIns.get("/zhsf/laws?msg=" + params.msg + "&page=" + params.page + "&page_size=" + params.page_size)
}

// 案例详情跳转法规
export const jumpLawsData = (params: any) => {
    return abcIns.get("/zhsf/lawdetail/bytitle?law_title=" + params)
}
// 获取智能问答问题推荐
export const getAIQuestion = (params: any) => {
    if (params.keyword) {
        return abcIns.get("/zhsf/qacard/question?keyword=" + params.keyword)
    } else {
        return abcIns.get("/zhsf/qacard/question");
    }
}
// 获取智能问答问题答案
export const getAIAnswer = (params: any) => {
    return abcIns.get("/zhsf/qacard/answer?question_id=" + params.id)
}
//获取类案检索报告接口
export const getCaseReportData = (params: any) => {
    return abcIns.post('/zhsf/report/graph', params)
}
//获取类案检索报告接口（新增的接口：不包含法律法规）
export const getCaseReportOne = (params: any) => {
    return abcIns.post('/zhsf/report/case_es ', params)
}
//获取类案检索报告接口（新增的接口：只包含法律法规）
export const getCaseReportTwo = (params: any) => {
    return abcIns.post('/zhsf/report/case_law ', params)
}
// 增加推荐词的计数
export const addQuestionPullListValue = (params: any) => {
    return abcIns.get("/zhsf/visitadd?question_id=" + params)
}
// 案由检索
export const anyou = (params: any) => {
    return abcIns.get("/zhsf/anyou")
}
// 获取问答列表数据
export const getQuesList = (params: any) => {
    if (params.query) {
        return abcIns.get("/zhsf/user_questions?is_resolved=" + params.is_resolved + "&anyou_type=" + params.anyou_type + "&page=" + params.page + "&page_size=" + params.page_size + "&days=" + params.days + "&msg=" + params.msg)
    } else {
        return abcIns.get("/zhsf/user_questions?is_resolved=" + params.is_resolved + "&anyou_type=" + params.anyou_type + "&page=" + params.page + "&page_size=" + params.page_size + "&days=" + params.days)
    }
}
// 获取问题详情内容
export const getQuestionDetail = (params: any) => {
    return abcIns.get("/zhsf/question_detail?question_id=" + params.id + "&source=" + params.source);
}

export const getInfo = (params: any) => {
    console.log('aaaaaaaaa')
    return abcIns.get("/zhsf/question_detail?question_id=" + params.id + "&source=" + params.source);
}
