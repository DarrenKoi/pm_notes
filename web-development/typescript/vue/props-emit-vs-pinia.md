---
tags: [vue, state-management, props, emit, pinia]
level: intermediate
last_updated: 2026-02-01
status: complete
---

# Props/Emit vs Pinia: Vue 상태 관리 패턴 선택 가이드

> 컴포넌트 간 데이터 전달에 Props/Emit과 Pinia를 언제 사용해야 하는지, 그리고 왜 Pinia를 모든 곳에 쓰면 안 되는지 정리한다.

## 왜 필요한가? (Why)

Vue 앱이 커지면 "이 상태를 어디서 관리할까?"라는 질문이 반복된다. 많은 입문자가 Pinia(전역 상태 관리)를 배우고 나면 **모든 상태를 Pinia에 넣는** 실수를 한다.

이렇게 하면 발생하는 문제:

- **컴포넌트 재사용성 파괴**: Pinia store에 의존하는 컴포넌트는 해당 store 없이 동작할 수 없다
- **데이터 흐름 추적 어려움**: props/emit은 부모-자식 관계가 명시적이지만, store는 어디서든 읽고 쓸 수 있어 흐름이 불투명해진다
- **테스트 복잡도 증가**: 단순한 컴포넌트 테스트에도 store mocking이 필요해진다
- **불필요한 전역 상태 증가**: 로컬에서만 쓰이는 상태가 전역에 노출된다

**핵심 원칙**: 상태는 가능한 한 **가장 좁은 범위**에서 관리한다.

## 핵심 개념 (What)

### 1. Props (부모 → 자식 데이터 전달)

부모가 자식에게 데이터를 내려주는 **단방향 바인딩**. 자식은 props를 직접 수정할 수 없다.

```vue
<!-- Parent.vue -->
<ChildComponent :userName="user.name" :isActive="true" />
```

```vue
<!-- ChildComponent.vue -->
<script setup lang="ts">
defineProps<{
  userName: string
  isActive: boolean
}>()
</script>
```

### 2. Emit (자식 → 부모 이벤트 전달)

자식이 부모에게 **"이런 일이 일어났다"**고 알리는 방법. 부모가 이벤트를 받아 처리한다.

```vue
<!-- ChildComponent.vue -->
<script setup lang="ts">
const emit = defineEmits<{
  update: [newName: string]
  delete: []
}>()

function handleClick() {
  emit('update', 'new value')
}
</script>
```

```vue
<!-- Parent.vue -->
<ChildComponent @update="handleUpdate" @delete="handleDelete" />
```

### 3. Pinia (전역 상태 관리)

앱 전체에서 **여러 컴포넌트가 공유해야 하는 상태**를 관리한다.

```ts
// stores/auth.ts
export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const isLoggedIn = computed(() => !!user.value)

  async function login(credentials: Credentials) {
    user.value = await api.login(credentials)
  }

  return { user, isLoggedIn, login }
})
```

## 어떻게 사용하는가? (How)

### 판단 기준 플로우차트

```
이 상태를 다른 컴포넌트가 알아야 하는가?
├─ No → 컴포넌트 로컬 state (ref/reactive)
└─ Yes → 부모-자식 관계인가?
    ├─ Yes → Props/Emit
    └─ No → 형제 컴포넌트이거나 깊은 트리인가?
        ├─ 2~3단계 → Props/Emit (또는 provide/inject)
        └─ 여러 페이지/라우트에서 공유 → Pinia
```

### 같은 시나리오, 두 가지 구현 비교

**시나리오**: 유저 목록에서 유저를 선택하면 상세 정보를 보여준다.

#### Props/Emit으로 구현 (권장)

```vue
<!-- UserPage.vue -->
<script setup lang="ts">
import { ref } from 'vue'

interface User {
  id: number
  name: string
  email: string
}

const users = ref<User[]>([
  { id: 1, name: '김철수', email: 'kim@example.com' },
  { id: 2, name: '이영희', email: 'lee@example.com' },
])
const selectedUser = ref<User | null>(null)

function selectUser(user: User) {
  selectedUser.value = user
}
</script>

<template>
  <div class="user-page">
    <UserList :users="users" @select="selectUser" />
    <UserDetail v-if="selectedUser" :user="selectedUser" />
  </div>
</template>
```

```vue
<!-- UserList.vue -->
<script setup lang="ts">
defineProps<{ users: User[] }>()
const emit = defineEmits<{ select: [user: User] }>()
</script>

<template>
  <ul>
    <li v-for="user in users" :key="user.id" @click="emit('select', user)">
      {{ user.name }}
    </li>
  </ul>
</template>
```

```vue
<!-- UserDetail.vue -->
<script setup lang="ts">
defineProps<{ user: User }>()
</script>

<template>
  <div>
    <h2>{{ user.name }}</h2>
    <p>{{ user.email }}</p>
  </div>
</template>
```

**장점**: `UserList`와 `UserDetail`은 **어떤 데이터든** 받아서 표시할 수 있다. 재사용 가능.

#### Pinia로 구현 (이 경우 비권장)

```ts
// stores/userSelection.ts
export const useUserSelectionStore = defineStore('userSelection', () => {
  const users = ref<User[]>([])
  const selectedUser = ref<User | null>(null)

  function selectUser(user: User) {
    selectedUser.value = user
  }

  return { users, selectedUser, selectUser }
})
```

```vue
<!-- UserList.vue -->
<script setup lang="ts">
const store = useUserSelectionStore()
</script>

<template>
  <ul>
    <li v-for="user in store.users" :key="user.id" @click="store.selectUser(user)">
      {{ user.name }}
    </li>
  </ul>
</template>
```

**문제점**:
- `UserList`가 `useUserSelectionStore`에 **하드코딩 의존** → 다른 곳에서 재사용 불가
- 부모-자식 사이의 단순한 데이터 흐름을 전역 store로 우회 → 불필요한 복잡성
- 테스트 시 store mocking 필요

### Pinia를 써야 하는 경우

| 상황 | 예시 |
|------|------|
| 인증/세션 상태 | 로그인 유저 정보, 토큰 |
| 앱 전역 설정 | 다크모드, 언어 설정 |
| 여러 페이지에서 공유하는 데이터 | 장바구니, 알림 목록 |
| 캐싱이 필요한 API 응답 | 자주 참조하는 마스터 데이터 |

### Pinia를 쓰면 안 되는 경우

| 상황 | 대안 |
|------|------|
| 부모-자식 간 데이터 전달 | Props/Emit |
| 폼 입력 상태 | 컴포넌트 로컬 `ref` |
| UI 토글 (모달 열림/닫힘) | 컴포넌트 로컬 `ref` |
| 리스트에서 선택된 항목 | Props/Emit 또는 `v-model` |
| 2~3단계 깊이의 데이터 전달 | `provide`/`inject` |

### 안티패턴: "Pinia 만능주의"

```ts
// ❌ 이런 store를 만들지 말 것
export const useModalStore = defineStore('modal', () => {
  const isOpen = ref(false)
  const toggle = () => { isOpen.value = !isOpen.value }
  return { isOpen, toggle }
})
```

모달의 열림/닫힘은 해당 컴포넌트나 부모에서 `ref(false)`로 충분하다. 전역 store로 만들면:
- 앱 어디서든 모달을 열 수 있어 **의도치 않은 부작용** 발생 가능
- 모달이 여러 개면 store도 여러 개 필요 → store 난립

### 실무 가이드라인 요약

1. **기본값은 로컬 state** → `ref()`, `reactive()`
2. **부모-자식은 Props/Emit** → 명시적 데이터 흐름
3. **깊은 트리는 provide/inject** → prop drilling 방지
4. **진짜 전역 상태만 Pinia** → 인증, 설정, 공유 캐시

## 참고 자료 (References)

- [Vue 공식 문서 - Props](https://vuejs.org/guide/components/props.html)
- [Vue 공식 문서 - Events](https://vuejs.org/guide/components/events.html)
- [Pinia 공식 문서](https://pinia.vuejs.org/)
- [Vue 공식 문서 - State Management](https://vuejs.org/guide/scaling-up/state-management.html)

## 관련 문서

- [Vue README](./README.md)
