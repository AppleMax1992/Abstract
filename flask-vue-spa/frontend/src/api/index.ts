// @ts-ignore
import * as Fingerprint2 from '@fingerprintjs/fingerprintjs';


// 获取浏览器唯一标识
export const getBrowserKey = ()=>{
    return new Promise((resolve, reject)=>{
        Fingerprint2.getV18({}, function (data: any,components: any) {
            resolve(data);
          })
    })
}



