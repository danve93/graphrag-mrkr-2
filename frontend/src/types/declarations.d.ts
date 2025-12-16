declare module 'react-cytoscapejs' {
    import cytoscape from 'cytoscape'
    import { Component } from 'react'

    interface CytoscapeComponentProps {
        elements: cytoscape.ElementDefinition[]
        style?: React.CSSProperties
        stylesheet?: cytoscape.Stylesheet[]
        layout?: cytoscape.LayoutOptions
        cy?: (cy: cytoscape.Core) => void
        wheelSensitivity?: number
        className?: string
        zoom?: number
        pan?: { x: number; y: number }
        minZoom?: number
        maxZoom?: number
        zoomingEnabled?: boolean
        userZoomingEnabled?: boolean
        panningEnabled?: boolean
        userPanningEnabled?: boolean
        boxSelectionEnabled?: boolean
        autoungrabify?: boolean
        autounselectify?: boolean
    }

    export default class CytoscapeComponent extends Component<CytoscapeComponentProps> {
        static normalizeElements(elements: any): cytoscape.ElementDefinition[]
    }
}

declare module 'cytoscape-fcose'
