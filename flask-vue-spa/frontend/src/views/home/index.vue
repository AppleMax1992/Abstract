
<template>
  <div>
    <h1>摘要生成</h1>
    <div style="width:50%;float:left">
      <a-textarea :rows="30" @change="getInfo" placeholder="我是文本" :maxlength="1000" />
    </div>
    <div style="width:50%;float:right">
      <a-textarea :rows="30" v-model:value="abstract" placeholder="我是摘要" :maxlength="100" />
    </div>
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

const abstract = ref('')

const getInfo = () => {
  const path = `http://localhost:5000/api/getInfo`
  axios.get(path)
    .then(response => {
      console.log(response.data)
      abstract.value = response.data.getInfo
    })
    .catch(error => {
      console.log(error)
    })
}

// onMounted(() => {
//   getInfo()
// });
</script>
