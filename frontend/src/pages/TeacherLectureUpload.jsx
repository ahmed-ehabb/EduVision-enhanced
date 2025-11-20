import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import Sidebar from '../components/Layout/Sidebar'
import Header from '../components/Layout/Header'
import FileUpload from '../components/Common/FileUpload'
import { lectureService } from '../services/lectureService'
import toast from 'react-hot-toast'
import './TeacherLectureUpload.css'

export default function TeacherLectureUpload() {
  const navigate = useNavigate()
  const [audioFile, setAudioFile] = useState(null)
  const [textbookFile, setTextbookFile] = useState(null)
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    course_code: '',
    course_name: '',
    chapter_start: '',
    chapter_end: '',
  })
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)

  const handleInputChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()

    if (!audioFile) {
      toast.error('Please upload an audio/video file')
      return
    }

    if (!formData.title.trim()) {
      toast.error('Please enter a lecture title')
      return
    }

    try {
      setUploading(true)
      setUploadProgress(0)

      // Create FormData
      const data = new FormData()
      data.append('audio_file', audioFile)
      data.append('title', formData.title)

      if (formData.description) data.append('description', formData.description)
      if (formData.course_code) data.append('course_code', formData.course_code)
      if (formData.course_name) data.append('course_name', formData.course_name)

      if (textbookFile) {
        data.append('textbook_file', textbookFile)
        if (formData.chapter_start) data.append('chapter_start_anchor', formData.chapter_start)
        if (formData.chapter_end) data.append('chapter_end_anchor', formData.chapter_end)
      }

      // Upload lecture
      const result = await lectureService.uploadLecture(data)

      toast.success('Lecture uploaded successfully!')

      // Navigate to processing status page
      navigate(`/teacher/lectures/${result.job_id}/status`)

    } catch (error) {
      console.error('Upload failed:', error)
      toast.error(error.message || 'Failed to upload lecture')
    } finally {
      setUploading(false)
    }
  }

  return (
    <div className="dashboard-layout">
      <Sidebar role="teacher" />
      <div className="dashboard-main">
        <Header
          title="Upload Lecture"
          subtitle="Upload audio/video and textbook for AI processing"
        />

        <div className="dashboard-content">
          <div className="upload-container">
            <form onSubmit={handleSubmit}>
              {/* Basic Information */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Basic Information</h3>
                  <p className="card-subtitle">Enter lecture details</p>
                </div>
                <div className="card-body">
                  <div className="form-group">
                    <label className="form-label">Lecture Title *</label>
                    <input
                      type="text"
                      name="title"
                      className="form-input"
                      value={formData.title}
                      onChange={handleInputChange}
                      placeholder="Introduction to Machine Learning"
                      required
                      disabled={uploading}
                    />
                  </div>

                  <div className="form-group">
                    <label className="form-label">Description</label>
                    <textarea
                      name="description"
                      className="form-input"
                      value={formData.description}
                      onChange={handleInputChange}
                      placeholder="Brief description of the lecture content..."
                      rows="3"
                      disabled={uploading}
                    />
                  </div>

                  <div className="grid grid-2">
                    <div className="form-group">
                      <label className="form-label">Course Code</label>
                      <input
                        type="text"
                        name="course_code"
                        className="form-input"
                        value={formData.course_code}
                        onChange={handleInputChange}
                        placeholder="CS401"
                        disabled={uploading}
                      />
                    </div>
                    <div className="form-group">
                      <label className="form-label">Course Name</label>
                      <input
                        type="text"
                        name="course_name"
                        className="form-input"
                        value={formData.course_name}
                        onChange={handleInputChange}
                        placeholder="Machine Learning"
                        disabled={uploading}
                      />
                    </div>
                  </div>
                </div>
              </div>

              {/* Audio/Video Upload */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Lecture Audio/Video *</h3>
                  <p className="card-subtitle">Upload your lecture recording</p>
                </div>
                <div className="card-body">
                  <FileUpload
                    label="Audio/Video File"
                    helperText="Drag and drop your lecture audio or video file"
                    accept={{
                      'audio/*': ['.mp3', '.wav', '.m4a', '.ogg', '.flac'],
                      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
                    }}
                    maxSize={500 * 1024 * 1024} // 500MB
                    onFileSelect={setAudioFile}
                    disabled={uploading}
                  />
                  <p className="form-help">
                    Supported formats: MP3, WAV, M4A, MP4, MOV. Maximum size: 500MB
                  </p>
                </div>
              </div>

              {/* Optional Textbook */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Textbook PDF (Optional)</h3>
                  <p className="card-subtitle">Upload textbook for content alignment analysis</p>
                </div>
                <div className="card-body">
                  <FileUpload
                    label="Textbook PDF"
                    helperText="Drag and drop your textbook PDF file"
                    accept={{
                      'application/pdf': ['.pdf'],
                    }}
                    maxSize={100 * 1024 * 1024} // 100MB
                    onFileSelect={setTextbookFile}
                    disabled={uploading}
                  />

                  {textbookFile && (
                    <div style={{ marginTop: 'var(--spacing-lg)' }}>
                      <div className="grid grid-2">
                        <div className="form-group">
                          <label className="form-label">Chapter Start Anchor</label>
                          <input
                            type="text"
                            name="chapter_start"
                            className="form-input"
                            value={formData.chapter_start}
                            onChange={handleInputChange}
                            placeholder="e.g., 'Chapter 3: Neural Networks'"
                            disabled={uploading}
                          />
                          <p className="form-help">Text to identify chapter start</p>
                        </div>
                        <div className="form-group">
                          <label className="form-label">Chapter End Anchor</label>
                          <input
                            type="text"
                            name="chapter_end"
                            className="form-input"
                            value={formData.chapter_end}
                            onChange={handleInputChange}
                            placeholder="e.g., 'Chapter 4: Deep Learning'"
                            disabled={uploading}
                          />
                          <p className="form-help">Text to identify chapter end</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Submit Button */}
              <div className="upload-actions">
                <button
                  type="button"
                  className="btn btn-outline"
                  onClick={() => navigate('/teacher/lectures')}
                  disabled={uploading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="btn btn-primary btn-lg"
                  disabled={uploading || !audioFile}
                >
                  {uploading ? (
                    <>
                      <div className="spinner spinner-sm"></div>
                      Uploading... {uploadProgress}%
                    </>
                  ) : (
                    <>
                      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="7 10 12 15 17 10" />
                        <line x1="12" y1="15" x2="12" y2="3" />
                      </svg>
                      Upload & Process
                    </>
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  )
}
