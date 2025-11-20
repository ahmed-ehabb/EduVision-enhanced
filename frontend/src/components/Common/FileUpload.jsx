import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import './FileUpload.css'

export default function FileUpload({
  onFileSelect,
  accept = {},
  maxSize = 500 * 1024 * 1024, // 500MB default
  label = "Upload File",
  helperText = "Drag and drop or click to browse",
  icon,
  disabled = false
}) {
  const [error, setError] = useState(null)

  const onDrop = useCallback((acceptedFiles, rejectedFiles) => {
    setError(null)

    if (rejectedFiles.length > 0) {
      const rejection = rejectedFiles[0]
      if (rejection.errors[0]?.code === 'file-too-large') {
        setError(`File is too large. Maximum size is ${maxSize / (1024 * 1024)}MB`)
      } else if (rejection.errors[0]?.code === 'file-invalid-type') {
        setError('Invalid file type')
      } else {
        setError('File upload failed')
      }
      return
    }

    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0])
    }
  }, [onFileSelect, maxSize])

  const { getRootProps, getInputProps, isDragActive, acceptedFiles } = useDropzone({
    onDrop,
    accept,
    maxSize,
    multiple: false,
    disabled,
  })

  const file = acceptedFiles[0]

  return (
    <div className="file-upload-container">
      <label className="file-upload-label">{label}</label>

      <div
        {...getRootProps()}
        className={`file-upload-dropzone ${isDragActive ? 'drag-active' : ''} ${disabled ? 'disabled' : ''} ${error ? 'has-error' : ''}`}
      >
        <input {...getInputProps()} />

        {icon || (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        )}

        {file ? (
          <div className="file-selected">
            <div className="file-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z" />
                <polyline points="13 2 13 9 20 9" />
              </svg>
            </div>
            <div className="file-info">
              <div className="file-name">{file.name}</div>
              <div className="file-size">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
            </div>
            <button
              type="button"
              className="file-remove"
              onClick={(e) => {
                e.stopPropagation()
                acceptedFiles.splice(0, 1)
                onFileSelect(null)
              }}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <line x1="18" y1="6" x2="6" y2="18" />
                <line x1="6" y1="6" x2="18" y2="18" />
              </svg>
            </button>
          </div>
        ) : (
          <div className="file-upload-prompt">
            <p className="prompt-text">
              {isDragActive ? 'Drop file here' : helperText}
            </p>
            <p className="prompt-hint">Maximum file size: {maxSize / (1024 * 1024)}MB</p>
          </div>
        )}
      </div>

      {error && (
        <div className="file-upload-error">
          {error}
        </div>
      )}
    </div>
  )
}
