/*
 * @Description: 
 * @Author: gaozhimei
 * @Date: 2021-11-25 15:08:43
 */
/* eslint-disable */
declare module '*.vue' {
  import type { DefineComponent } from 'vue'
  const component: DefineComponent<{}, {}, any>
  export default component
}
