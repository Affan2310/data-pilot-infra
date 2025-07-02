import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os
import base64
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional
import json

class ComputerVisionAnalyzer:
    """Computer vision analysis for defect detection and image processing."""
    
    def __init__(self):
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        self.analysis_results = {}
    
    def analyze_image_defects(self, image_data: bytes, filename: str) -> Dict[str, Any]:
        """
        Analyze image for defects using various computer vision techniques.
        """
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Basic image properties
            height, width, channels = img.shape
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            # Perform various defect detection analyses
            results = {
                'filename': filename,
                'image_properties': {
                    'width': int(width),
                    'height': int(height),
                    'channels': int(channels),
                    'total_pixels': int(width * height)
                },
                'defect_analysis': {},
                'quality_metrics': {},
                'anomaly_detection': {}
            }
            
            # 1. Edge detection for structural defects
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (width * height)
            
            results['defect_analysis']['edge_density'] = float(edge_density)
            results['defect_analysis']['edge_interpretation'] = self._interpret_edge_density(edge_density)
            
            # 2. Contour analysis for shape anomalies
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            contour_stats = {
                'total_contours': len(contours),
                'large_contours': len([c for c in contours if cv2.contourArea(c) > 100]),
                'average_contour_area': np.mean([cv2.contourArea(c) for c in contours]) if contours else 0
            }
            
            results['defect_analysis']['contour_analysis'] = contour_stats
            
            # 3. Color analysis for discoloration defects
            color_stats = self._analyze_color_distribution(img, hsv)
            results['defect_analysis']['color_analysis'] = color_stats
            
            # 4. Texture analysis using statistical measures
            texture_stats = self._analyze_texture(gray)
            results['defect_analysis']['texture_analysis'] = texture_stats
            
            # 5. Brightness and contrast analysis
            brightness_contrast = self._analyze_brightness_contrast(gray)
            results['quality_metrics'] = brightness_contrast
            
            # 6. Noise detection
            noise_level = self._detect_noise(gray)
            results['quality_metrics']['noise_level'] = noise_level
            
            # 7. Blur detection
            blur_score = self._detect_blur(gray)
            results['quality_metrics']['blur_score'] = blur_score
            
            # 8. Overall defect score
            defect_score = self._calculate_defect_score(results)
            results['anomaly_detection']['overall_defect_score'] = defect_score
            results['anomaly_detection']['classification'] = self._classify_defect_level(defect_score)
            
            # 9. Generate recommendations
            recommendations = self._generate_recommendations(results)
            results['recommendations'] = recommendations
            
            return results
            
        except Exception as e:
            return {
                'filename': filename,
                'error': f"Analysis failed: {str(e)}",
                'defect_analysis': {},
                'quality_metrics': {},
                'anomaly_detection': {}
            }
    
    def analyze_multiple_images(self, image_files: List[Tuple[bytes, str]]) -> Dict[str, Any]:
        """Analyze multiple images and provide comparative defect analysis."""
        results = []
        
        for image_data, filename in image_files:
            result = self.analyze_image_defects(image_data, filename)
            results.append(result)
        
        # Comparative analysis
        if len(results) > 1:
            comparative_stats = self._generate_comparative_analysis(results)
        else:
            comparative_stats = {}
        
        return {
            'individual_results': results,
            'comparative_analysis': comparative_stats,
            'summary': self._generate_batch_summary(results)
        }
    
    def _interpret_edge_density(self, edge_density: float) -> str:
        """Interpret edge density for defect assessment."""
        if edge_density < 0.01:
            return "Very low edge density - possible blur or smooth surface"
        elif edge_density < 0.03:
            return "Low edge density - normal smooth surface"
        elif edge_density < 0.08:
            return "Moderate edge density - normal textured surface"
        elif edge_density < 0.15:
            return "High edge density - complex texture or potential surface irregularities"
        else:
            return "Very high edge density - possible noise, cracks, or severe surface defects"
    
    def _analyze_color_distribution(self, img: np.ndarray, hsv: np.ndarray) -> Dict[str, Any]:
        """Analyze color distribution for defect detection."""
        # Calculate color statistics
        bgr_means = np.mean(img, axis=(0, 1))
        bgr_stds = np.std(img, axis=(0, 1))
        
        # HSV statistics
        hsv_means = np.mean(hsv, axis=(0, 1))
        hsv_stds = np.std(hsv, axis=(0, 1))
        
        # Color uniformity (lower std indicates more uniform color)
        color_uniformity = 1.0 / (1.0 + np.mean(bgr_stds))
        
        # Detect potential discoloration by analyzing color variance
        color_variance = np.var(img, axis=(0, 1))
        
        return {
            'bgr_means': [float(x) for x in bgr_means],
            'bgr_stds': [float(x) for x in bgr_stds],
            'hsv_means': [float(x) for x in hsv_means],
            'color_uniformity': float(color_uniformity),
            'color_variance': [float(x) for x in color_variance],
            'potential_discoloration': bool(np.any(color_variance > 2000))
        }
    
    def _analyze_texture(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze texture patterns for surface defect detection."""
        # Calculate texture statistics using Local Binary Patterns approach
        # Simplified version using gradient analysis
        
        # Sobel operators for gradient analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Texture statistics
        texture_mean = np.mean(gradient_magnitude)
        texture_std = np.std(gradient_magnitude)
        texture_uniformity = 1.0 / (1.0 + texture_std)
        
        # Detect texture anomalies
        texture_threshold = texture_mean + 2 * texture_std
        anomaly_pixels = np.sum(gradient_magnitude > texture_threshold)
        anomaly_percentage = (anomaly_pixels / gradient_magnitude.size) * 100
        
        return {
            'gradient_mean': float(texture_mean),
            'gradient_std': float(texture_std),
            'texture_uniformity': float(texture_uniformity),
            'anomaly_percentage': float(anomaly_percentage),
            'has_texture_anomalies': bool(anomaly_percentage > 5.0)
        }
    
    def _analyze_brightness_contrast(self, gray: np.ndarray) -> Dict[str, Any]:
        """Analyze brightness and contrast for quality assessment."""
        # Basic statistics
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Contrast using RMS contrast
        contrast_rms = np.sqrt(np.mean((gray - mean_brightness) ** 2))
        
        # Dynamic range
        min_val, max_val = np.min(gray), np.max(gray)
        dynamic_range = max_val - min_val
        
        # Assess quality
        brightness_quality = self._assess_brightness_quality(mean_brightness)
        contrast_quality = self._assess_contrast_quality(contrast_rms)
        
        return {
            'mean_brightness': float(mean_brightness),
            'brightness_std': float(brightness_std),
            'contrast_rms': float(contrast_rms),
            'dynamic_range': float(dynamic_range),
            'brightness_quality': brightness_quality,
            'contrast_quality': contrast_quality
        }
    
    def _detect_noise(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect noise levels in the image."""
        # Use Laplacian for noise detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_score = np.var(laplacian)
        
        # Classify noise level
        if noise_score < 100:
            noise_level = "Low"
        elif noise_score < 500:
            noise_level = "Moderate"
        else:
            noise_level = "High"
        
        return {
            'noise_score': float(noise_score),
            'noise_level': noise_level
        }
    
    def _detect_blur(self, gray: np.ndarray) -> Dict[str, Any]:
        """Detect blur using Laplacian variance."""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Blur classification
        if laplacian_var < 100:
            blur_level = "Severe blur detected"
        elif laplacian_var < 500:
            blur_level = "Moderate blur detected"
        else:
            blur_level = "Sharp image"
        
        return {
            'laplacian_variance': float(laplacian_var),
            'blur_assessment': blur_level,
            'is_blurred': bool(laplacian_var < 500)
        }
    
    def _calculate_defect_score(self, results: Dict[str, Any]) -> float:
        """Calculate overall defect score (0-100, higher = more defects)."""
        score = 0.0
        
        # Edge density contribution
        edge_density = results['defect_analysis'].get('edge_density', 0)
        if edge_density > 0.15:
            score += 25
        elif edge_density > 0.08:
            score += 10
        
        # Color analysis contribution
        color_analysis = results['defect_analysis'].get('color_analysis', {})
        if color_analysis.get('potential_discoloration', False):
            score += 20
        
        # Texture analysis contribution
        texture_analysis = results['defect_analysis'].get('texture_analysis', {})
        if texture_analysis.get('has_texture_anomalies', False):
            score += 20
        
        # Quality metrics contribution
        quality_metrics = results.get('quality_metrics', {})
        noise_level = quality_metrics.get('noise_level', {}).get('noise_level', 'Low')
        if noise_level == 'High':
            score += 15
        elif noise_level == 'Moderate':
            score += 10
        
        blur_info = quality_metrics.get('blur_score', {})
        if blur_info.get('is_blurred', False):
            score += 15
        
        return min(score, 100.0)
    
    def _classify_defect_level(self, defect_score: float) -> str:
        """Classify defect level based on score."""
        if defect_score < 20:
            return "Excellent - No significant defects detected"
        elif defect_score < 40:
            return "Good - Minor defects detected"
        elif defect_score < 60:
            return "Fair - Moderate defects detected"
        elif defect_score < 80:
            return "Poor - Significant defects detected"
        else:
            return "Critical - Severe defects detected"
    
    def _assess_brightness_quality(self, brightness: float) -> str:
        """Assess brightness quality."""
        if brightness < 50:
            return "Too dark"
        elif brightness < 100:
            return "Slightly dark"
        elif brightness < 180:
            return "Good brightness"
        elif brightness < 220:
            return "Slightly bright"
        else:
            return "Too bright"
    
    def _assess_contrast_quality(self, contrast: float) -> str:
        """Assess contrast quality."""
        if contrast < 20:
            return "Very low contrast"
        elif contrast < 40:
            return "Low contrast"
        elif contrast < 80:
            return "Good contrast"
        else:
            return "High contrast"
    
    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Check defect score
        defect_score = results['anomaly_detection'].get('overall_defect_score', 0)
        if defect_score > 60:
            recommendations.append("Further inspection recommended - significant defects detected")
        
        # Check brightness
        quality_metrics = results.get('quality_metrics', {})
        brightness_quality = quality_metrics.get('brightness_quality', '')
        if 'dark' in brightness_quality.lower():
            recommendations.append("Improve lighting conditions for better image quality")
        elif 'bright' in brightness_quality.lower():
            recommendations.append("Reduce lighting intensity to avoid overexposure")
        
        # Check blur
        blur_info = quality_metrics.get('blur_score', {})
        if blur_info.get('is_blurred', False):
            recommendations.append("Image appears blurred - check camera focus and stability")
        
        # Check noise
        noise_info = quality_metrics.get('noise_level', {})
        if noise_info.get('noise_level') == 'High':
            recommendations.append("High noise detected - check camera settings and environmental conditions")
        
        # Check texture anomalies
        texture_analysis = results['defect_analysis'].get('texture_analysis', {})
        if texture_analysis.get('has_texture_anomalies', False):
            recommendations.append("Surface texture irregularities detected - inspect for physical defects")
        
        # Check color issues
        color_analysis = results['defect_analysis'].get('color_analysis', {})
        if color_analysis.get('potential_discoloration', False):
            recommendations.append("Color variations detected - check for material inconsistencies")
        
        if not recommendations:
            recommendations.append("Image quality appears acceptable - no major issues detected")
        
        return recommendations
    
    def _generate_comparative_analysis(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comparative analysis across multiple images."""
        if not results:
            return {}
        
        # Extract defect scores
        defect_scores = [r['anomaly_detection'].get('overall_defect_score', 0) for r in results]
        
        # Calculate statistics
        comparative_stats = {
            'total_images': len(results),
            'average_defect_score': float(np.mean(defect_scores)),
            'max_defect_score': float(np.max(defect_scores)),
            'min_defect_score': float(np.min(defect_scores)),
            'std_defect_score': float(np.std(defect_scores)),
            'images_with_defects': len([s for s in defect_scores if s > 40]),
            'critical_images': len([s for s in defect_scores if s > 80])
        }
        
        # Find best and worst images
        best_idx = np.argmin(defect_scores)
        worst_idx = np.argmax(defect_scores)
        
        comparative_stats['best_quality_image'] = results[best_idx]['filename']
        comparative_stats['worst_quality_image'] = results[worst_idx]['filename']
        
        return comparative_stats
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary for batch analysis."""
        successful_analyses = [r for r in results if 'error' not in r]
        failed_analyses = [r for r in results if 'error' in r]
        
        summary = {
            'total_images': len(results),
            'successful_analyses': len(successful_analyses),
            'failed_analyses': len(failed_analyses),
        }
        
        if successful_analyses:
            defect_scores = [r['anomaly_detection'].get('overall_defect_score', 0) for r in successful_analyses]
            summary.update({
                'overall_quality_rating': self._classify_defect_level(np.mean(defect_scores)),
                'defect_distribution': {
                    'excellent': len([s for s in defect_scores if s < 20]),
                    'good': len([s for s in defect_scores if 20 <= s < 40]),
                    'fair': len([s for s in defect_scores if 40 <= s < 60]),
                    'poor': len([s for s in defect_scores if 60 <= s < 80]),
                    'critical': len([s for s in defect_scores if s >= 80])
                }
            })
        
        return summary
    
    def create_annotated_image(self, image_data: bytes, analysis_results: Dict[str, Any]) -> bytes:
        """Create annotated image with defect detection overlay."""
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Create overlay for annotations
            overlay = img.copy()
            
            # Add defect score annotation
            defect_score = analysis_results['anomaly_detection'].get('overall_defect_score', 0)
            classification = analysis_results['anomaly_detection'].get('classification', 'Unknown')
            
            # Choose color based on defect level
            if defect_score < 20:
                color = (0, 255, 0)  # Green
            elif defect_score < 40:
                color = (0, 255, 255)  # Yellow
            elif defect_score < 60:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 0, 255)  # Red
            
            # Add text annotations
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(overlay, f"Defect Score: {defect_score:.1f}", (10, 30), font, 0.8, color, 2)
            cv2.putText(overlay, f"Status: {classification.split(' - ')[0]}", (10, 60), font, 0.8, color, 2)
            
            # Encode back to bytes
            _, buffer = cv2.imencode('.jpg', overlay)
            return buffer.tobytes()
            
        except Exception as e:
            return image_data  # Return original if annotation fails
        
    def generate_human_friendly_report(self, results: Dict[str, Any]) -> str:
        try:
            lines = [
                f"📄 Defect Analysis Report for `{results.get('filename', 'Unknown')}`",
                "",
                "🖼 Image Properties:",
                f"- Width: {results['image_properties'].get('width')} px",
                f"- Height: {results['image_properties'].get('height')} px",
                f"- Channels: {results['image_properties'].get('channels')}",
                f"- Total Pixels: {results['image_properties'].get('total_pixels')}",
                "",
                "⚙️ Defect Analysis:",
                f"- Edge Density: {results['defect_analysis'].get('edge_density'):.3f}",
                f"- Edge Interpretation: {results['defect_analysis'].get('edge_interpretation')}",
                f"- Total Contours: {results['defect_analysis']['contour_analysis'].get('total_contours')}",
                f"- Large Contours: {results['defect_analysis']['contour_analysis'].get('large_contours')}",
                f"- Avg Contour Area: {results['defect_analysis']['contour_analysis'].get('average_contour_area'):.2f}",
                f"- Potential Discoloration: {results['defect_analysis']['color_analysis'].get('potential_discoloration')}",
                "",
                "🌞 Quality Metrics:",
                f"- Brightness: {results['quality_metrics'].get('mean_brightness'):.2f} ({results['quality_metrics'].get('brightness_quality')})",
                f"- Contrast: {results['quality_metrics'].get('contrast_quality')}",
                f"- Noise Level: {results['quality_metrics']['noise_level'].get('noise_level')}",
                f"- Blur: {results['quality_metrics']['blur_score'].get('blur_assessment')}",
                "",
                "🚨 Anomaly Detection:",
                f"- Defect Score: {results['anomaly_detection'].get('overall_defect_score')}",
                f"- Classification: {results['anomaly_detection'].get('classification')}",
                "",
                "💡 Recommendations:"
            ]
            for rec in results.get('recommendations', []):
                lines.append(f"✅ {rec}")
            return '\n'.join(lines)
        except Exception as e:
            return f"Error generating report: {str(e)}"
