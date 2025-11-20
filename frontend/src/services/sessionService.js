import api from './api'

export const sessionService = {
  async createSession(lectureId, scheduledAt = null) {
    const response = await api.post('/api/sessions', {
      lecture_id: lectureId,
      scheduled_at: scheduledAt,
    })
    return response.data
  },

  async startSession(sessionCode) {
    const response = await api.post(`/api/sessions/${sessionCode}/start`)
    return response.data
  },

  async endSession(sessionCode) {
    const response = await api.post(`/api/sessions/${sessionCode}/end`)
    return response.data
  },

  async joinSession(sessionCode) {
    const response = await api.post(`/api/sessions/${sessionCode}/join`)
    return response.data
  },

  async logEngagement(sessionId, events) {
    const response = await api.post(`/api/sessions/${sessionId}/engagement`, events)
    return response.data
  },

  async getAnalytics(sessionId) {
    const response = await api.get(`/api/sessions/${sessionId}/analytics`)
    return response.data
  },
}
