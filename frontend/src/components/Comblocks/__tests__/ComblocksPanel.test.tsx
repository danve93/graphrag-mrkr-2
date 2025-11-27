import React from 'react'
import { render, screen } from '@testing-library/react'
import ComblocksPanel from '../ComblocksPanel'

describe('ComblocksPanel', () => {
  it('renders heading and placeholder content', () => {
    render(<ComblocksPanel />)

    expect(screen.getByText(/Comblocks \(test panel\)/i)).toBeInTheDocument()
    expect(screen.getByText(/Drag items here to create/i)).toBeInTheDocument()
  })
})
