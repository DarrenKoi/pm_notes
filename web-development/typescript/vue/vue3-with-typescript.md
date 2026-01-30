---
tags: [vue, typescript, frontend, composition-api]
level: beginner
last_updated: 2026-01-31
status: complete
---

# Vue 3 + TypeScript 시작하기

> Vue 3의 Composition API와 TypeScript를 함께 사용하여 타입 안전한 프론트엔드를 만드는 방법

## 왜 필요한가? (Why)

- Vue 3는 TypeScript로 재작성되어 **1급 TypeScript 지원** 제공
- props, emit, ref 등에 타입을 지정하면 **IDE 자동 완성과 컴파일 타임 에러 검출** 가능
- Flask 백엔드의 API 응답 타입을 프론트엔드에서 그대로 사용하면 **풀스택 타입 일관성** 확보
- Composition API + TypeScript 조합이 현재 Vue 생태계의 표준

## 프로젝트 셋업 (How)

### 1. 프로젝트 생성

```bash
# Vite + Vue + TypeScript (추천)
npm create vite@latest my-app -- --template vue-ts
cd my-app
npm install
npm run dev
```

생성되는 구조:

```
my-app/
├── src/
│   ├── App.vue
│   ├── main.ts              # .js가 아닌 .ts
│   ├── components/
│   │   └── HelloWorld.vue
│   └── vite-env.d.ts        # Vite 타입 선언
├── tsconfig.json
├── tsconfig.app.json
├── vite.config.ts
└── package.json
```

### 2. Vue 파일에서 TypeScript 사용

Vue 단일 파일 컴포넌트(SFC)에서 `<script setup lang="ts">`를 선언하면 TypeScript가 활성화된다.

```vue
<script setup lang="ts">
import { ref, computed } from 'vue'

// ref는 자동으로 타입 추론됨
const count = ref(0)          // Ref<number>
const name = ref('Daeyoung')  // Ref<string>

// 명시적 타입 지정도 가능
const items = ref<string[]>([])

// computed도 반환 타입 자동 추론
const doubled = computed(() => count.value * 2)  // ComputedRef<number>

function increment() {
  count.value++
}
</script>

<template>
  <button @click="increment">{{ count }}</button>
  <p>Doubled: {{ doubled }}</p>
</template>
```

---

## 핵심 패턴

### 1. 인터페이스 정의와 활용

API 응답이나 데이터 모델을 인터페이스로 정의하면 앱 전체에서 재사용할 수 있다.

```typescript
// src/types/index.ts
export interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'user' | 'guest'
}

export interface ApiResponse<T> {
  data: T
  message: string
  success: boolean
}
```

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'
import type { User, ApiResponse } from '@/types'

const users = ref<User[]>([])
const loading = ref(false)

async function fetchUsers() {
  loading.value = true
  const res = await fetch('/api/users')
  const json: ApiResponse<User[]> = await res.json()
  users.value = json.data
  loading.value = false
}

onMounted(fetchUsers)
</script>

<template>
  <div v-if="loading">Loading...</div>
  <ul v-else>
    <li v-for="user in users" :key="user.id">
      {{ user.name }} ({{ user.role }})
    </li>
  </ul>
</template>
```

### 2. Props 타입 정의

```vue
<script setup lang="ts">
// 방법 1: defineProps에 타입 파라미터 사용 (추천)
const props = defineProps<{
  title: string
  count?: number              // optional
  items: string[]
  status: 'active' | 'inactive'
}>()

// 방법 2: 기본값이 필요한 경우 withDefaults 사용
const props2 = withDefaults(defineProps<{
  title: string
  count: number
  variant: 'primary' | 'secondary'
}>(), {
  count: 0,
  variant: 'primary'
})
</script>

<template>
  <h1>{{ title }}</h1>
  <span>Count: {{ count }}</span>
</template>
```

부모 컴포넌트에서 사용 시 잘못된 타입을 넘기면 **빌드 타임에 에러**가 발생한다:

```vue
<!-- 부모 컴포넌트 -->
<template>
  <!-- ✅ 정상 -->
  <MyComponent title="Hello" :count="5" :items="['a']" status="active" />

  <!-- ❌ 타입 에러: status에 'unknown'은 불가 -->
  <MyComponent title="Hello" :items="[]" status="unknown" />
</template>
```

### 3. Emit 타입 정의

```vue
<script setup lang="ts">
// emit 이벤트와 페이로드에 타입 지정
const emit = defineEmits<{
  (e: 'update', id: number): void
  (e: 'delete', id: number): void
  (e: 'search', query: string): void
}>()

// Vue 3.3+ 간결한 문법
const emit2 = defineEmits<{
  update: [id: number]
  delete: [id: number]
  search: [query: string]
}>()

function handleClick(id: number) {
  emit('update', id)    // ✅ 타입 체크됨
  // emit('update', 'abc')  // ❌ 에러: string은 number에 할당 불가
}
</script>
```

### 4. Reactive 객체

```vue
<script setup lang="ts">
import { reactive } from 'vue'

// 인터페이스로 타입 지정
interface FormState {
  username: string
  email: string
  age: number | null
}

const form = reactive<FormState>({
  username: '',
  email: '',
  age: null
})

function submitForm() {
  // form.username은 string으로 타입 추론됨
  console.log(form.username.toUpperCase())  // ✅ 안전
}
</script>
```

### 5. Template Ref (DOM 접근)

```vue
<script setup lang="ts">
import { ref, onMounted } from 'vue'

// HTML 엘리먼트 ref
const inputEl = ref<HTMLInputElement | null>(null)

// 컴포넌트 ref
import MyComponent from './MyComponent.vue'
const compRef = ref<InstanceType<typeof MyComponent> | null>(null)

onMounted(() => {
  // null 체크 필요
  inputEl.value?.focus()
})
</script>

<template>
  <input ref="inputEl" />
  <MyComponent ref="compRef" />
</template>
```

### 6. Composable (재사용 로직)

Flask API와 통신하는 composable 예시:

```typescript
// src/composables/useApi.ts
import { ref } from 'vue'
import type { Ref } from 'vue'

interface UseApiReturn<T> {
  data: Ref<T | null>
  error: Ref<string | null>
  loading: Ref<boolean>
  execute: () => Promise<void>
}

export function useApi<T>(url: string): UseApiReturn<T> {
  const data = ref<T | null>(null) as Ref<T | null>
  const error = ref<string | null>(null)
  const loading = ref(false)

  async function execute() {
    loading.value = true
    error.value = null
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      data.value = await res.json()
    } catch (e) {
      error.value = e instanceof Error ? e.message : 'Unknown error'
    } finally {
      loading.value = false
    }
  }

  return { data, error, loading, execute }
}
```

컴포넌트에서 사용:

```vue
<script setup lang="ts">
import { onMounted } from 'vue'
import { useApi } from '@/composables/useApi'
import type { User } from '@/types'

// T가 User[]로 추론 → data는 Ref<User[] | null>
const { data: users, loading, error, execute } = useApi<User[]>('/api/users')

onMounted(execute)
</script>

<template>
  <div v-if="loading">Loading...</div>
  <div v-else-if="error">Error: {{ error }}</div>
  <ul v-else-if="users">
    <li v-for="user in users" :key="user.id">{{ user.name }}</li>
  </ul>
</template>
```

---

## Flask 백엔드와 연동할 때의 패턴

Flask API의 응답 구조에 맞춰 프론트엔드 타입을 정의하면 풀스택에서 타입 일관성을 유지할 수 있다.

```python
# Flask 백엔드 (app.py)
@app.route('/api/users')
def get_users():
    return jsonify({
        "data": [{"id": 1, "name": "Kim", "email": "kim@test.com", "role": "admin"}],
        "message": "success",
        "success": True
    })
```

```typescript
// Vue 프론트엔드 — 동일한 구조의 타입 정의
interface User {
  id: number
  name: string
  email: string
  role: 'admin' | 'user' | 'guest'
}

interface ApiResponse<T> {
  data: T
  message: string
  success: boolean
}
```

개발 시 Vite의 프록시를 활용하면 CORS 없이 Flask와 연동 가능:

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:5000',  // Flask 서버
        changeOrigin: true,
      }
    }
  }
})
```

---

## TypeScript를 Vue에서 쓸 때 자주 하는 실수

### 1. ref 값 접근 시 `.value` 빠뜨림

```typescript
const count = ref(0)
// ❌ script에서는 .value 필요
console.log(count)        // Ref 객체 자체가 출력됨
// ✅
console.log(count.value)  // 0

// template에서는 자동 unwrap되므로 .value 불필요
// <template>{{ count }}</template>  ← OK
```

### 2. reactive에 재할당

```typescript
const state = reactive({ name: 'Kim' })
// ❌ 반응성 소실
// state = { name: 'Lee' }
// ✅ 속성을 변경
state.name = 'Lee'
```

### 3. 타입 단언 남용

```typescript
// ❌ as로 강제 캐스팅하면 타입 안전성 깨짐
const user = {} as User

// ✅ 올바른 초기값 또는 null 사용
const user = ref<User | null>(null)
```

---

## 참고 자료

- [Vue 공식: TypeScript 가이드](https://vuejs.org/guide/typescript/overview.html)
- [Vue 공식: Composition API](https://vuejs.org/guide/extras/composition-api-faq.html)
- [Vite 공식 문서](https://vite.dev/guide/)

## 관련 문서

- [TypeScript 웹 개발 로드맵](../README.md)
- [Flask 관련 문서](../../python/flask/) (백엔드 연동)
