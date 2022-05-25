# flask-vue-spa
Vue.js SPA served over Flask microframework

* Python: 3.6.3
* Vue.js: 3.5.2
* vue-router: 3.0.1
* axios: 0.16.2

Tutorial on how I build this app:
https://medium.com/@oleg.agapov/full-stack-single-page-application-with-vue-js-and-flask-b1e036315532

## Build Setup

``` bash
# install front-end
cd frontend
yarn install

# serve with hot reload at localhost:8080
yarn serve

# build for production/Flask with minification
npm run build


# serve back-end at localhost:5000
FLASK_APP=run.py flask run
```

