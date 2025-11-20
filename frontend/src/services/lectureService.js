import axios from 'axios'

// Teacher Module API runs on port 8001
const TEACHER_API_URL = import.meta.env.VITE_TEACHER_API_URL || 'http://localhost:8001'

// Create axios instance for teacher API
const teacherApi = axios.create({
  baseURL: TEACHER_API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const lectureService = {
  /**
   * Upload lecture audio/video file with metadata
   * @param {FormData} formData - Contains audio_file, title, description, textbook_file (optional)
   * @returns {Promise<{job_id: string, message: string}>}
   */
  async uploadLecture(formData) {
    const response = await teacherApi.post('/api/lectures/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // Track upload progress
      onUploadProgress: (progressEvent) => {
        const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total)
        console.log(`Upload Progress: ${percentCompleted}%`)
      },
    })
    return response.data
  },

  /**
   * Get processing status for a lecture
   * @param {string} jobId - Job ID returned from upload
   * @returns {Promise<{status: string, progress: object, result: object}>}
   */
  async getProcessingStatus(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/status`)
    return response.data
  },

  /**
   * Get lecture transcript
   * @param {string} jobId - Job ID
   * @returns {Promise<{transcript: string, language: string}>}
   */
  async getTranscript(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/transcript`)
    return response.data
  },

  /**
   * Update lecture transcript
   * @param {string} jobId - Job ID
   * @param {string} transcript - Updated transcript text
   * @returns {Promise}
   */
  async updateTranscript(jobId, transcript) {
    const response = await teacherApi.put(`/api/lectures/${jobId}/transcript`, {
      transcript,
    })
    return response.data
  },

  /**
   * Get generated notes
   * @param {string} jobId - Job ID
   * @returns {Promise<{notes: string, format: string}>}
   */
  async getNotes(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/notes`)
    return response.data
  },

  /**
   * Get generated quiz questions
   * @param {string} jobId - Job ID
   * @returns {Promise<{questions: array}>}
   */
  async getQuiz(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/quiz`)
    return response.data
  },

  /**
   * Update quiz questions
   * @param {string} jobId - Job ID
   * @param {array} questions - Updated questions array
   * @returns {Promise}
   */
  async updateQuiz(jobId, questions) {
    const response = await teacherApi.put(`/api/lectures/${jobId}/quiz`, {
      questions,
    })
    return response.data
  },

  /**
   * Get content alignment analysis
   * @param {string} jobId - Job ID
   * @returns {Promise<{alignment_score: number, matches: array}>}
   */
  async getContentAlignment(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/alignment`)
    return response.data
  },

  /**
   * Get engagement analysis
   * @param {string} jobId - Job ID
   * @returns {Promise<{engagement_score: number, feedback: string}>}
   */
  async getEngagementAnalysis(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/engagement`)
    return response.data
  },

  /**
   * Get complete lecture result
   * @param {string} jobId - Job ID
   * @returns {Promise<object>} Complete processing result
   */
  async getResult(jobId) {
    const response = await teacherApi.get(`/api/lectures/${jobId}/result`)
    return response.data
  },

  /**
   * Check Teacher Module API health
   * @returns {Promise<{status: string}>}
   */
  async checkHealth() {
    const response = await teacherApi.get('/health')
    return response.data
  },

  /**
   * Get API status and statistics
   * @returns {Promise<object>}
   */
  async getStatus() {
    const response = await teacherApi.get('/api/status')
    return response.data
  },
}

// Response interceptor for error handling
teacherApi.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response) {
      // Server responded with error status
      const errorMessage = error.response.data?.detail || error.response.data?.message || error.message
      console.error('Teacher API Error:', errorMessage)
      throw new Error(errorMessage)
    } else if (error.request) {
      // Request made but no response
      console.error('Teacher API not responding')
      throw new Error('Teacher Module API is not responding. Make sure it is running on port 8000.')
    } else {
      // Something else happened
      console.error('Request Error:', error.message)
      throw error
    }
  }
)

export default lectureService
