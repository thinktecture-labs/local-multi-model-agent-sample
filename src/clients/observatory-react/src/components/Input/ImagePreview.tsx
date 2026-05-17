import { useAppState, useAppDispatch } from '../../state/AppContext.tsx'

export function ImagePreview() {
  const { pendingImageDataUrls } = useAppState()
  const dispatch = useAppDispatch()

  if (pendingImageDataUrls.length === 0) return null

  return (
    <div className="image-preview">
      {pendingImageDataUrls.map((url, idx) => (
        <div key={idx} className="image-preview-item">
          <img src={url} alt={`Pending upload ${idx + 1}`} className="image-preview-thumb" />
          <button
            className="image-preview-remove"
            onClick={() => dispatch({ type: 'REMOVE_PENDING_IMAGE', idx })}
            aria-label="Remove image"
          >
            &times;
          </button>
        </div>
      ))}
    </div>
  )
}
