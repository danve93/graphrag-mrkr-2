import React from 'react'

export default {
  title: 'Design/TokenButtons',
}

export const ButtonVariants = () => (
  <div style={{ display: 'flex', gap: 12, alignItems: 'center', padding: 16 }}>
    <button className="button-primary">Primary</button>
    <button className="small-button small-button-primary">Small Primary</button>
    <button className="small-button small-button-secondary">Small Secondary</button>
  </div>
)

export const ToggleExample = () => {
  const [on, setOn] = React.useState(true)
  return (
    <div style={{ padding: 16 }}>
      <div style={{ marginBottom: 8 }}>Toggle (click to change)</div>
      <button
        onClick={() => setOn((v) => !v)}
        className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${on ? 'toggle-on' : 'toggle-off'}`}
        aria-pressed={on}
      >
        <span className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${on ? 'translate-x-6' : 'translate-x-1'}`} />
      </button>
    </div>
  )
}
