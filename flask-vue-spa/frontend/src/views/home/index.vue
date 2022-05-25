
<template>
  <div>
    <h1>摘要生成</h1>
    <a-spin tip="Loading..." :spinning="loading">
      <div style="width:50%;float:left">
        <a-textarea :rows="30" @change="getInfo" placeholder="我是文本" :maxlength="1000" />
      </div>
      <div style="width:50%;float:right">
        <a-textarea :rows="30" v-model:value="abstract" placeholder="我是摘要" :maxlength="100" />
      </div>
    </a-spin>
  </div>
</template>

<script setup >
import {
  onBeforeMount,
  onMounted,
  reactive,
  ref,
  onBeforeUnmount,
  watch,
  nextTick,
} from "vue";
import axios from 'axios'
import * as api from "../../api";
const abstract = ref('')
let loading = ref(false);
const getInfo = (e) => {
  loading.value = true
  console.log(e.target.value)
  const content = e.target.value
  // let res = await api.getInfo(content);
  const path = `http://localhost:5000/api/getT5`
  axios.get(path + '?content=' + content)
    .then(response => {

      console.log(response.data)
      abstract.value = response.data.getInfo
      loading.value = false
    })
    .catch(error => {
      console.log(error)
    })
}

// onMounted(() => {
//   getInfo()
// });
</script>
