'use client'

import { useEffect, useRef, useState, useCallback } from 'react'

// MediaPipe types
interface HandLandmark {
    x: number
    y: number
    z: number
}

interface HandResults {
    multiHandLandmarks?: HandLandmark[][]
    multiHandedness?: Array<{ label: string }>
}

export type GestureMode = 'cursor' | 'rotate' | 'zoom' | 'none'

export interface GestureState {
    /** Cursor position in normalized screen coords (0-1) */
    cursor: { x: number; y: number }
    /** Rotation delta (X = yaw, Y = pitch) */
    rotation: { x: number; y: number }
    /** Zoom delta (1 = no zoom, <1 = zoom out, >1 = zoom in) */
    zoomDelta: number
    /** Is user making pinch gesture (click) */
    isPinching: boolean
    /** Pinch position is SAME as cursor position in this mode */
    pinchPosition: { x: number; y: number } | null
    /** Number of hands detected */
    handCount: number
    /** Active mode based on hand configuration */
    mode: GestureMode
    /** Is camera active and tracking */
    isTracking: boolean
    /** Camera permission status */
    permissionStatus: 'pending' | 'granted' | 'denied'
    /** Current gesture name for display */
    gestureName: string
}

// Thresholds
const PINCH_THRESHOLD = 0.05
const ZOOM_SENSITIVITY = 2.5
const ROTATION_SENSITIVITY = 3.0
const CURSOR_SMOOTHING = 0.5 // Heavy smoothing for specific pointing
const FIST_THRESHOLD = 0.1 // Finger tip to wrist/palm distance for "curled"

function distance2D(a: HandLandmark, b: HandLandmark): number {
    return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
}

function palmCenter(landmarks: HandLandmark[]): { x: number; y: number } {
    return {
        x: (landmarks[0].x + landmarks[9].x) / 2,
        y: (landmarks[0].y + landmarks[9].y) / 2,
    }
}

// Detect if hand is a closed fist
function isFist(landmarks: HandLandmark[]): boolean {
    const wrist = landmarks[0]
    // Check fingertips 8, 12, 16, 20 distance to wrist
    // If all are close to wrist, it's a fist
    const tips = [8, 12, 16, 20]
    let curledCount = 0
    tips.forEach(idx => {
        if (distance2D(landmarks[idx], wrist) < 0.3) { // 0.3 is rough normalized size of closed hand
            // Better check: Is tip below PIP joint? (y > y_pip for upright hand)
            // Simple geometry check: tip distance to wrist < pip distance to wrist?
            const tip = landmarks[idx]
            const mcp = landmarks[idx - 3] // Knuckle
            // If tip is closer to wrist than knuckle is, it's curled
            if (distance2D(tip, wrist) < distance2D(mcp, wrist)) {
                curledCount++
            }
        }
    })
    return curledCount >= 3 // Allow one finger slightly loose (like thumb)
}

// Detect pinch (Thumb + Index touching)
function isPinch(landmarks: HandLandmark[]): boolean {
    const thumbTip = landmarks[4]
    const indexTip = landmarks[8]
    return distance2D(thumbTip, indexTip) < 0.05 // 0.05 threshold
}

export function useGestureControls(enabled: boolean = true) {
    const [gestureState, setGestureState] = useState<GestureState>({
        cursor: { x: 0.5, y: 0.5 },
        rotation: { x: 0, y: 0 },
        zoomDelta: 1,
        isPinching: false,
        pinchPosition: null,
        handCount: 0,
        mode: 'none',
        isTracking: false,
        permissionStatus: 'pending',
        gestureName: '',
    })

    const videoRef = useRef<HTMLVideoElement | null>(null)
    const canvasRef = useRef<HTMLCanvasElement | null>(null)
    const handsRef = useRef<any>(null)
    const cameraRef = useRef<any>(null)
    const drawingUtilsRef = useRef<any>(null)

    // Tracking state
    const prevPalmPositions = useRef<Array<{ x: number; y: number }>>([])
    const prevHandDistance = useRef<number | null>(null)
    const smoothedCursor = useRef({ x: 0.5, y: 0.5 })

    // Process hand landmarks
    const processResults = useCallback((results: HandResults) => {
        const landmarks = results.multiHandLandmarks
        const handCount = landmarks?.length || 0
        const canvas = canvasRef.current
        const video = videoRef.current

        // --- Debug Draw ---
        if (canvas && video && landmarks) {
            const ctx = canvas.getContext('2d')
            if (ctx) {
                ctx.save()
                ctx.clearRect(0, 0, canvas.width, canvas.height)
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
                if (drawingUtilsRef.current) {
                    const { drawConnectors, drawLandmarks, HAND_CONNECTIONS } = drawingUtilsRef.current
                    for (const lms of landmarks) {
                        drawConnectors(ctx, lms, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 2 })
                        drawLandmarks(ctx, lms, { color: '#FF0000', lineWidth: 1, radius: 2 })
                    }
                }
                ctx.restore()
            }
        }

        if (!landmarks || handCount === 0) {
            setGestureState(prev => ({
                ...prev,
                handCount: 0,
                mode: 'none',
                gestureName: 'No hands detected',
                isPinching: false,
                rotation: { x: 0, y: 0 },
                zoomDelta: 1,
            }))
            prevPalmPositions.current = []
            return
        }

        const currentPalmPositions = landmarks.map(palmCenter)

        let mode: GestureMode = 'none'
        let gestureName = 'Tracking...'
        let cursor = smoothedCursor.current
        let isPinching = false
        let rotation = { x: 0, y: 0 }
        let zoomDelta = 1

        // --- GESTURE LOGIC ---

        // CASE 1: Single Hand -> CURSOR MODE
        if (handCount === 1) {
            mode = 'cursor'
            const hand = landmarks[0]

            // Map index finger tip (8) to cursor position
            // Flip X because webcam is mirrored
            const rawX = 1 - hand[8].x
            const rawY = hand[8].y

            // Smooth cursor
            cursor = {
                x: smoothedCursor.current.x * CURSOR_SMOOTHING + rawX * (1 - CURSOR_SMOOTHING),
                y: smoothedCursor.current.y * CURSOR_SMOOTHING + rawY * (1 - CURSOR_SMOOTHING)
            }
            smoothedCursor.current = cursor

            // Check Pinch (Thumb 4 + Index 8)
            const pinchDist = distance2D(hand[4], hand[8])
            if (pinchDist < PINCH_THRESHOLD) {
                isPinching = true
                gestureName = '👌 Click'
            } else {
                gestureName = '✋ Cursor'
            }
        }

        // CASE 2: Two Hands -> ROTATE (Fists) or ZOOM (Pinches)
        else if (handCount === 2) {
            const hand1 = landmarks[0]
            const hand2 = landmarks[1]
            const hand1Fist = isFist(hand1)
            const hand2Fist = isFist(hand2)
            const hand1Pinch = isPinch(hand1)
            const hand2Pinch = isPinch(hand2)

            // Sub-case 2.1: Both Fists -> ROTATION
            if (hand1Fist && hand2Fist) {
                mode = 'rotate'
                gestureName = '✊ Rotate'

                // Track mid-point
                const p1 = currentPalmPositions[0]
                const p2 = currentPalmPositions[1]
                const midX = (p1.x + p2.x) / 2
                const midY = (p1.y + p2.y) / 2

                // Calc rotation delta
                if (prevPalmPositions.current.length === 2) {
                    const prevP1 = prevPalmPositions.current[0]
                    const prevP2 = prevPalmPositions.current[1]
                    const prevMidX = (prevP1.x + prevP2.x) / 2
                    const prevMidY = (prevP1.y + prevP2.y) / 2

                    rotation = {
                        x: (midX - prevMidX) * ROTATION_SENSITIVITY,
                        y: (midY - prevMidY) * ROTATION_SENSITIVITY
                    }
                }
            }

            // Sub-case 2.2: Both Pinches -> ZOOM
            else if (hand1Pinch && hand2Pinch) {
                mode = 'zoom'

                const p1 = currentPalmPositions[0]
                const p2 = currentPalmPositions[1]
                const dist = Math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

                if (prevHandDistance.current !== null) {
                    const delta = dist - prevHandDistance.current
                    if (Math.abs(delta) > 0.002) {
                        zoomDelta = 1 + delta * ZOOM_SENSITIVITY
                        gestureName = delta > 0 ? '👐 Zoom In' : '🙏 Zoom Out'
                    }
                }
                prevHandDistance.current = dist
            }

            // Sub-case 2.3: Open/Other -> IDLE (Safe State)
            else {
                mode = 'none'
                gestureName = '👐 Idle'
                prevHandDistance.current = null // Reset zoom tracking
            }
        }

        prevPalmPositions.current = currentPalmPositions

        setGestureState({
            cursor,
            rotation,
            zoomDelta,
            isPinching,
            pinchPosition: isPinching ? cursor : null,
            handCount,
            mode,
            isTracking: true,
            permissionStatus: 'granted',
            gestureName,
        })

    }, [])

    // Initialize MediaPipe (Same as before)
    useEffect(() => {
        if (!enabled) return
        let mounted = true

        async function init() {
            try {
                const { Hands, HAND_CONNECTIONS } = await import('@mediapipe/hands')
                const { Camera } = await import('@mediapipe/camera_utils')
                const drawingUtils = await import('@mediapipe/drawing_utils')

                if (!mounted) return
                drawingUtilsRef.current = { ...drawingUtils, HAND_CONNECTIONS }

                const video = document.createElement('video')
                video.style.display = 'none'
                video.setAttribute('playsinline', '')
                document.body.appendChild(video)
                videoRef.current = video

                const canvas = document.createElement('canvas')
                canvas.width = 320
                canvas.height = 240
                document.body.appendChild(canvas)
                canvasRef.current = canvas

                const hands = new Hands({
                    locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
                })

                hands.setOptions({
                    maxNumHands: 2,
                    modelComplexity: 1,
                    minDetectionConfidence: 0.6,
                    minTrackingConfidence: 0.5,
                })

                hands.onResults((results: HandResults) => {
                    if (mounted) processResults(results)
                })

                handsRef.current = hands

                const camera = new Camera(video, {
                    onFrame: async () => {
                        if (handsRef.current) await handsRef.current.send({ image: video })
                    },
                    width: 320,
                    height: 240,
                })

                cameraRef.current = camera
                await camera.start()

                if (mounted) {
                    setGestureState(prev => ({ ...prev, permissionStatus: 'granted', isTracking: true }))
                }
            } catch (error) {
                console.error('Gesture init failed:', error)
                if (mounted) setGestureState(prev => ({ ...prev, permissionStatus: 'denied' }))
            }
        }

        init()

        return () => {
            mounted = false
            cameraRef.current?.stop()
            videoRef.current?.remove()
            canvasRef.current?.remove()
        }
    }, [enabled, processResults])

    const getVideoElement = useCallback(() => canvasRef.current, [])
    return { gestureState, getVideoElement }
}
