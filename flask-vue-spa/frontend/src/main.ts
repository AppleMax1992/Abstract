/*
 * @Description: 
 * @Author: gaozhimei
 * @Date: 2021-11-29 10:11:33
 */
import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import Antd from 'ant-design-vue'
import 'ant-design-vue/dist/antd.css'
import * as echarts from 'echarts';
let app = createApp(App);
app.use(router).use(Antd).mount('#app');
