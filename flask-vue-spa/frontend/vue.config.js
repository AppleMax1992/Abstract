/*
 * @Description: 
 * @Author: gaozhimei
 * @Date: 2021-11-25 15:09:44
 */
module.exports = {
    devServer: {
        // host: 'localhost',
        // port: 8080,
        //proxy:{'/api':{}},代理器中设置/api,项目中请求路径为/api的替换为target
        proxy: {
            '/abc': {
                // target: 'http://58.213.45.42:25007', //代理地址，这里设置的地址会代替axios中设置的baseURL
                // target: 'http://81.70.9.146:25007',
                target: 'http://192.168.3.4:25007',
                changeOrigin: true, // 如果接口跨域，需要进行这个参数配置
                pathRewrite: {
                    '^/abc': '/'
                } //pathRewrite方法重写url
                //pathRewrite: {'^/api': '/'} 重写之后url为 http://192.168.1.16:8085/xxxx
                //pathRewrite: {'^/api': '/api'} 重写之后url为 http://192.168.1.16:8085/api/xxxx
            },
            '/hwy': {
                // target: 'http://192.168.10.148:8000',
                target: 'http://58.213.45.42:25015',
                // target: 'http://81.70.9.146:25015',
                // target: 'http://192.168.3.65:25015',
                changeOrigin: true,
                pathRewrite: {
                    '^/hwy': '/'
                }
            }
        }
    },

    //添加web worker loader
    configureWebpack: config => {
        config.module.rules.push({
            test: /\.worker.js$/,
            use: {
                loader: 'worker-loader',
                options: { inline: true, name: 'workerName.[hash].js' }
            }
        })
    },
    // parallel: require('os').cpus().length > 1,
    parallel: false,
}