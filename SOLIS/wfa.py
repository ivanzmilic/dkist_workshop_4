import numpy as np
import scipy.integrate

####################################################################################################
def calc_lam0_cog( si, wave, lam0_nom ):
    #
    # Return an estimate of the central line-core wavelength of an input
    # Stokes I absorption line spectrum
    #
    #   @param   :  si    :  [float nparray(*)]  :  Observed Stokes I absorption spectrum
    #   @param   :  wave  :  [float nparray(*)]  :  Observed wavelengths array
    #
    #   @return  :  lam0  :  [float]             :  central line-core wavelength
    #

    wcore    = np.where( abs(wave - lam0_nom) <= 0.4 )
    si_resid = np.max( si ) - si
    numer    = scipy.integrate.simps( si_resid[wcore]*wave[wcore], wave[wcore] )
    denom    = scipy.integrate.simps( si_resid[wcore], wave[wcore] )
    if numer != 0.:
        try:
            lam0 = numer / denom
        except ZeroDivisionError:
            lam0 = 8542.0910    # default to lab wavelength
    else:
        lam0 = 8542.0910

    return lam0
####################################################################################################



####################################################################################################
def calc_lam0_vzc( sv, wave, lam0_nom ):
    #
    # Return an estimate of the central line-core wavelength of an input
    # Stokes V net circular polarization spectrum, based on the zero-crossing
    # position of the Stokes V spectrum
    #
    #   @param   :  sv    :  [float nparray(*)]  :  Observed Stokes V absorption spectrum
    #   @param   :  wave  :  [float nparray(*)]  :  Observed wavelengths array
    #
    #   @return  :  lam0  :  [float]             :  central line-core wavelength
    #

    wcore = np.where( abs(wave - lam0_nom) <= 0.2 )[0]
    for i in range(0,len(wcore)):
        idx = wcore[0] + i
        if sv[idx]*sv[idx+1] < 0.:
            m = (sv[idx+1] - sv[idx]) / (wave[idx+1] - wave[idx])
            lam0 = wave[idx] - sv[idx]/m
            break

    try:
        lam0
    except NameError:
        lam0 = 8542.0910

    return lam0
####################################################################################################



####################################################################################################
def wfa_blos( data, wave, dsi_dlam, c_los ):
    #
    # Calculate the LOS field strength from the Weak Field Approximation
    #
    #   @param   :  data      :  [float nparray(ns,nw)]  :  Stokes profile data for a single pixel
    #   @param   :  wave      :  [float nparray(nw)]     :  Observed wavelengths [Ang]       
    #   @param   :  dsi_dlam  :  [float nparray(nw)]     :  Derivative of Stokes I profile w.r.t.
    #                                                       wavelength
    #   @param   :  c_los     :  [float]                 :  Scaling factor for LOS field
    #
    #   @return  :  blos      :  [float]                 :  LOS field strength [G]
    #

    try:
        # linear regression
        blos = -np.sum( dsi_dlam*data[3,:] )/( c_los*np.sum( dsi_dlam**2 ) )
    except ZeroDivisionError:
        blos = 0.

    return blos
####################################################################################################



####################################################################################################
def wfa_btrn( data, wave, dsi_dlam, c_trn, lc ):
    #
    # Calculate the transverse field strength from the Weak Field Approximation
    #
    #   @param   :  data      :  [float nparray(ns,nw)]  :  Stokes profile data for a single pixel
    #   @param   :  wave      :  [float nparray(nw)]     :  Observed wavelengths [Ang]        
    #   @param   :  dsi_dlam  :  [float nparray(nw)]     :  Derivative of Stokes I profile w.r.t.
    #                                                       wavelength
    #   @param   :  c_trn     :  [float]                 :  Scaling factor for transverse field
    #   @param   :  lc        :  [float]                 :  Observed line-center wavelength [Ang]
    #
    #   @return  :  btrn      :  [float]                 :  Transverse field strength [G]
    #

    sl_scaled  = (4./(3.*c_trn))*np.sqrt( data[1,:]**2 + data[2,:]**2 )
    inv_offset = 1./(wave - lc)
    product    = abs( inv_offset )*abs( dsi_dlam )
    try:
        # linear regression
        btrn = np.sqrt( np.sum( sl_scaled*product )/np.sum( product**2 ) )
    except ZeroDivisionError:
        btrn = 0.

    return btrn
####################################################################################################



####################################################################################################
def wfa_binc( blos, btrn ):
    #
    # Calculate the magnetic field inclination from the Weak Field Approximation
    #
    #   @param   :  blos  :  [float]  :  LOS field strength [G] for a single pixel
    #   @param   :  btrn  :  [float]  :  Transverse field strength [G] for a single pixel
    #
    #   @return  :  binc  :  [float]  :  Field inclination [deg]
    #

    binc = np.arctan2( btrn, blos )*(180./np.pi)

    return binc
####################################################################################################



####################################################################################################
def wfa_bazm( data ):
    #
    # Calculate the transverse magnetic field azimuthal angle from the Weak Field Approximation.
    # Note that the arctan2 function is used here to resolve the quadrant degeneracy automatically
    # (i.e. no requirement to check sign of Q and U at line-center to determine proper quadrant).
    #
    #   @param   :  data  :  [float nparray(ns,nw)]  :  Stokes profile data for a single pixel
    #
    #   @return  :  bazm  :  [float]                 :  Transverse field azimuth [deg]
    #

    bazm = 0.5*np.arctan2( np.sum( data[2,:] ), np.sum( data[1,:] ) )*(180./np.pi) + 90.    

    return bazm
####################################################################################################



####################################################################################################
def extract_range( data, wave, lc_dist, pos ):
    #
    # Extract the data in the appropriate spectral region from the input data, 
    # given the boundaries of the region (in input pos)
    #
    #   @param   :  data       :  [float nparray(ns,nw)]     :  Stokes I, Q, U, V data for a single
    #                                                           spatial pixel
    #   @param   :  wave       :  [float nparray(nw)]        :  Calibrated wavelength scale for
    #                                                           data [Ang]
    #   @param   :  lc_dis     :  [float nparray(nw)]        :  Array of wavelength distances from
    #                                                           line-center [Ang]
    #   @param   :  pos        :  [float nparray(2)]         :  Inner, outer wavelength positions for
    #                                                           region [Ang]
    #
    #   @return  :  count      :  [int]                      :  Number of points in region
    #   @return  :  wave_out   :  [float nparray(count)]     :  Observed wavelengths in region [Ang]
    #   @return  :  data_out   :  [float nparray(ns,count)]  :  Extracted data array for region
    #   @return  :  deriv_out  :  [float nparray(count)]     :  Stokes I derivative w.r.t.
    #                                                           wavelength in region
    #

    wrange = np.where( np.logical_and( lc_dist >= pos[0], lc_dist <= pos[1] ) )[0]
    count  = len( wrange )
    if count > 0:
        wave_out  = wave[wrange]
        data_out  = data[:,wrange]
        deriv_out = ( np.gradient( data[0,:], wave, edge_order=2 ) )[wrange]
    else:
        wave_out  = 0.
        data_out  = 0.
        deriv_out = 0.

    return count, wave_out, data_out, deriv_out
####################################################################################################



####################################################################################################
def wfapprox( data, wave, glos_eff, gtrn_eff, lam0 ):
    #
    # Wrapper function for calculating chromospheric and photospheric vector
    # magnetic fields using extracted spectral ranges
    #
    #   @param   :  data      :  [float nparray(ns,nw)]  :  Stokes I, Q, U, V data for a single
    #                                                       spatial pixel
    #   @param   :  wave      :  [float nparray(nw)]     :  Calibrated wavelength scale for
    #                                                       data [Ang]
    #   @param   :  glos_eff  :  [float]                 :  Effective Lande g-factor for LOS field
    #   @param   :  gtrn_eff  :  [float]                 :  Effective Lande g-factor for TRN field
    #   @param   :  lam0      :  [float]                 :  Observed line-center wavelength [Ang]
    #   
    #   @return  :  c_int     :  [float]                 :  Chromospheric (core) intensity
    #   @return  :  c_blos    :  [float]                 :  Chromospheric LOS field strength [G]
    #   @return  :  c_btrn    :  [float]                 :  Chromospheric transverse field strength [G]
    #   @return  :  c_bfld    :  [float]                 :  Chromospheric total field strength [G]
    #   @return  :  c_binc    :  [float]                 :  Chromospheric field inclination [deg]
    #   @return  :  c_bazm    :  [float]                 :  Chromospheric transverse field azimuth [deg]
    #   @return  :  p_int     :  [float]                 :  Photospheric (wing) intensity
    #   @return  :  p_blos    :  [float]                 :  Photospheric LOS field strength [G]
    #   @return  :  p_btrn    :  [float]                 :  Photospheric transverse field strength [G]
    #   @return  :  p_bfld    :  [float]                 :  Photospheric total field strength [G]
    #   @return  :  p_binc    :  [float]                 :  Photospheric field inclination [deg]
    #   @return  :  p_bazm    :  [float]                 :  Photospheric transverse field azimuth [deg]
    #

    #
    # define spectral ranges [in Ang] and other constants
    #
    corepos = [ 0.00, 0.25 ]
    cwbypos = [ 0.10, 0.40 ]
    wingpos = [ 0.75, 1.25 ]
    kfac    = 4.6686E-13*(lam0**2)
    c_los   = kfac*glos_eff
    c_trn   = (kfac**2)*gtrn_eff

    #
    # define distance from line-center [in Ang] and extract
    # required data from core, core-wing boundary, and
    # wing ranges 
    # 
    lc_dist = abs( wave - lam0 )
    core_count, core_wave, core_data, core_deriv = extract_range( data, wave, lc_dist, corepos )
    cwby_count, cwby_wave, cwby_data, cwby_deriv = extract_range( data, wave, lc_dist, cwbypos )
    wing_count, wing_wave, wing_data, wing_deriv = extract_range( data, wave, lc_dist, wingpos )
    if len( core_wave ) == 0 or len( cwby_wave ) == 0 or len( wing_wave ) == 0:
        print( "[ WARNING ] Zero-length spectral region(s)...continuing." )
        print( "[ WARNING ]     len( core_wave ) = ", len( core_wave ) )
        print( "[ WARNING ]     len( wing_wave ) = ", len( wing_wave ) )
        print( "[ WARNING ]     len( cwby_wave ) = ", len( cwby_wave ) )

    #
    # calculate chromospheric WFA using core and
    # core-wing boundary spectral range(s)
    #
    if core_count > 0 and cwby_count > 0:
        c_int  = np.min( core_data[0,:] )
        c_blos = wfa_blos( core_data, core_wave, core_deriv, c_los )
        c_btrn = wfa_btrn( cwby_data, cwby_wave, cwby_deriv, c_trn, lam0 )
        c_bfld = np.sqrt( c_blos**2 + c_btrn**2 )
        c_binc = wfa_binc( c_blos, c_btrn )
        c_bazm = wfa_bazm( core_data )
    else:
        c_int  = 0.
        c_blos = 0.
        c_btrn = 0.
        c_bfld = 0.
        c_binc = 0.
        c_bazm = 0.

    #
    # calculate photospheric WFA using wing
    # spectral range
    #
    if wing_count > 0:
        p_int  = np.mean( wing_data[0,:] )
        p_blos = wfa_blos( wing_data, wing_wave, wing_deriv, c_los )
        p_btrn = wfa_btrn( wing_data, wing_wave, wing_deriv, c_trn, lam0 )
        p_bfld = np.sqrt( p_blos**2 + p_btrn**2 )
        p_binc = wfa_binc( p_blos, p_btrn )
        p_bazm = wfa_bazm( wing_data )
    else:
        p_int  = 0.
        p_blos = 0.
        p_btrn = 0.
        p_bfld = 0.
        p_binc = 0.
        p_bazm = 0.

    return np.array( [ c_int, c_blos, c_btrn, c_bfld, c_binc, c_bazm, \
                       p_int, p_blos, p_btrn, p_bfld, p_binc, p_bazm ] )
####################################################################################################
