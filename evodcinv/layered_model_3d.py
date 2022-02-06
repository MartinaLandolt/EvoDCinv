# generate a collection of layered models indexed by cell

# invert(self, dispersion_dict_from_tomo, interfaces)
    # set x and y values based on cells available in both tomo and interfaces
    # set self.cells and cell.dcurves
        # attribute dcurves to cell in the same way as in the cross-section plot

# cost function
    # parameters : vp, vp/vs x n_layers
    # loop on self.cells
        # loop on cell.dcurves
            # propagate, pick, compute misfit
            # update cost function
            # store dispersion dict
    # compute average dispersion curve (phase for ex.)
    # compute misfit with respect to the array processing result
    # update cost function