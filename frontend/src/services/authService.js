import api from './api'

export const authService = {
  async login(email, password) {
    const response = await api.post('/auth/login', { email, password })
    const { access_token, refresh_token, user } = response.data

    localStorage.setItem('access_token', access_token)
    localStorage.setItem('refresh_token', refresh_token)
    localStorage.setItem('user', JSON.stringify(user))

    return response.data
  },

  async logout() {
    try {
      await api.post('/auth/logout')
    } finally {
      localStorage.clear()
    }
  },

  async register(userData) {
    const response = await api.post('/auth/register', userData)
    return response.data
  },

  getCurrentUser() {
    const userStr = localStorage.getItem('user')
    return userStr ? JSON.parse(userStr) : null
  },

  isAuthenticated() {
    return !!localStorage.getItem('access_token')
  },

  getRole() {
    const user = this.getCurrentUser()
    return user?.role
  },
}
